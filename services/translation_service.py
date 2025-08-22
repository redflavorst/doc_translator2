# services/translation_service.py
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# 기존 번역기 임포트
from quality_contract_translator import QualityContractTranslator

@dataclass
class TranslationResult:
    """번역 결과를 담는 데이터 클래스"""
    success: bool
    input_file: str
    output_file: str
    report: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None

@dataclass
class BatchTranslationResult:
    """배치 번역 결과"""
    success: bool
    results: List[TranslationResult]
    total_files: int
    successful_files: int
    failed_files: int
    average_confidence: float
    total_processing_time: float
    low_quality_files: List[str]

class TranslationService:
    """quality_contract_translator.py의 QualityContractTranslator를 서비스로 래핑"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 번역기 설정
        self.model_name = self.config.get('model_name', 'qwen3:8b')
        self.temperature = self.config.get('temperature', 0.1)
        self.max_retries = self.config.get('max_retries', 3)
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        self.quality_threshold = self.config.get('quality_threshold', 0.6)
        
        # 번역기 초기화
        self._translator = None
    
    def _get_translator(self) -> QualityContractTranslator:
        """번역기 인스턴스 획득 (싱글톤 패턴)"""
        if self._translator is None:
            try:
                self._translator = QualityContractTranslator(
                    model_name=self.model_name,
                    ollama_url=self.ollama_url,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    debug_raw_output=True
                )
                self.logger.info(f"번역기 초기화 완료: {self.model_name}")
            except Exception as e:
                self.logger.error(f"번역기 초기화 실패: {e}")
                raise RuntimeError(f"번역기 초기화 실패: {e}")
        
        return self._translator
    
    def translate_document(self, input_file: str, output_file: str = None) -> TranslationResult:
        """
        단일 마크다운 문서 번역
        
        Args:
            input_file: 번역할 마크다운 파일 경로
            output_file: 번역 결과를 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            TranslationResult: 번역 결과
        """
        start_time = time.time()
        
        try:
            # 입력 파일 검증
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")
            
            # 출력 파일 경로 설정
            if output_file is None:
                output_file = str(input_path.parent / f"{input_path.stem}_korean{input_path.suffix}")
            
            self.logger.info(f"문서 번역 시작: {input_path.name}")
            
            # 번역기 사용
            translator = self._get_translator()
            report = translator.translate_document_premium(
                input_file=str(input_file),
                output_file=output_file
            )
            
            processing_time = time.time() - start_time
            
            # 번역 성공 여부 확인
            success = (
                report.get('average_confidence', 0) >= self.quality_threshold and
                Path(output_file).exists()
            )
            
            self.logger.info(f"문서 번역 완료: {processing_time:.2f}초, 신뢰도: {report.get('average_confidence', 0):.2f}")
            
            return TranslationResult(
                success=success,
                input_file=str(input_file),
                output_file=output_file,
                report=report,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"문서 번역 실패: {e}")
            return TranslationResult(
                success=False,
                input_file=str(input_file),
                output_file=output_file or "",
                report={},
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def translate_documents_batch(self, markdown_files: List[str], output_dir: str = None) -> BatchTranslationResult:
        """
        여러 마크다운 파일 배치 번역
        
        Args:
            markdown_files: 번역할 마크다운 파일 경로 리스트
            output_dir: 번역 결과를 저장할 디렉토리 (None이면 각 파일과 같은 디렉토리)
            
        Returns:
            BatchTranslationResult: 배치 번역 결과
        """
        start_time = time.time()
        results = []
        successful_files = 0
        failed_files = 0
        total_confidence = 0.0
        confidence_count = 0
        low_quality_files = []
        
        self.logger.info(f"배치 번역 시작: {len(markdown_files)}개 파일")
        
        for i, md_file in enumerate(markdown_files, 1):
            self.logger.info(f"번역 진행 중 ({i}/{len(markdown_files)}): {Path(md_file).name}")
            
            try:
                # 출력 파일 경로 결정
                input_path = Path(md_file)
                if output_dir:
                    output_path = Path(output_dir) / f"{input_path.stem}_korean{input_path.suffix}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_file = str(output_path)
                else:
                    output_file = None
                
                # 개별 파일 번역
                result = self.translate_document(md_file, output_file)
                results.append(result)
                
                if result.success:
                    successful_files += 1
                    confidence = result.report.get('average_confidence', 0)
                    total_confidence += confidence
                    confidence_count += 1
                    
                    # 품질이 낮은 파일 기록
                    if confidence < self.quality_threshold:
                        low_quality_files.append(md_file)
                else:
                    failed_files += 1
                    
            except Exception as e:
                self.logger.error(f"파일 번역 중 예외 발생 ({md_file}): {e}")
                failed_files += 1
                results.append(TranslationResult(
                    success=False,
                    input_file=md_file,
                    output_file="",
                    report={},
                    processing_time=0,
                    error=str(e)
                ))
        
        # 전체 결과 계산
        total_processing_time = time.time() - start_time
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        batch_success = (failed_files == 0) and (len(low_quality_files) == 0)
        
        self.logger.info(f"배치 번역 완료: {successful_files}/{len(markdown_files)} 성공, "
                        f"평균 신뢰도: {average_confidence:.2f}, "
                        f"총 시간: {total_processing_time:.2f}초")
        
        return BatchTranslationResult(
            success=batch_success,
            results=results,
            total_files=len(markdown_files),
            successful_files=successful_files,
            failed_files=failed_files,
            average_confidence=average_confidence,
            total_processing_time=total_processing_time,
            low_quality_files=low_quality_files
        )
    
    def retry_failed_translations(self, batch_result: BatchTranslationResult) -> BatchTranslationResult:
        """
        실패한 번역 재시도
        
        Args:
            batch_result: 이전 배치 번역 결과
            
        Returns:
            BatchTranslationResult: 재시도 결과
        """
        failed_files = [r.input_file for r in batch_result.results if not r.success]
        
        if not failed_files:
            self.logger.info("재시도할 실패 파일이 없습니다.")
            return batch_result
        
        self.logger.info(f"실패한 번역 재시도: {len(failed_files)}개 파일")
        
        # 재시도 시 설정 조정 (온도 낮추기, 재시도 횟수 증가)
        original_config = self.config.copy()
        self.config['temperature'] = max(0.05, self.config.get('temperature', 0.1) - 0.05)
        self.config['max_retries'] = self.config.get('max_retries', 3) + 2
        
        # 번역기 재초기화
        self._translator = None
        
        try:
            retry_result = self.translate_documents_batch(failed_files)
            
            # 성공한 재시도 결과로 원본 결과 업데이트
            updated_results = []
            retry_dict = {r.input_file: r for r in retry_result.results}
            
            for original_result in batch_result.results:
                if original_result.input_file in retry_dict:
                    retry_res = retry_dict[original_result.input_file]
                    if retry_res.success:
                        updated_results.append(retry_res)
                    else:
                        updated_results.append(original_result)  # 여전히 실패
                else:
                    updated_results.append(original_result)  # 원래 성공했던 것
            
            # 통계 재계산
            successful = sum(1 for r in updated_results if r.success)
            failed = len(updated_results) - successful
            
            avg_conf = sum(r.report.get('average_confidence', 0) 
                          for r in updated_results if r.success) / max(successful, 1)
            
            low_quality = [r.input_file for r in updated_results 
                          if r.success and r.report.get('average_confidence', 0) < self.quality_threshold]
            
            return BatchTranslationResult(
                success=(failed == 0 and len(low_quality) == 0),
                results=updated_results,
                total_files=len(updated_results),
                successful_files=successful,
                failed_files=failed,
                average_confidence=avg_conf,
                total_processing_time=batch_result.total_processing_time + retry_result.total_processing_time,
                low_quality_files=low_quality
            )
            
        finally:
            # 원래 설정 복원
            self.config = original_config
            self._translator = None
    
    def get_translation_status(self, input_files: List[str], output_dir: str = None) -> Dict[str, Dict[str, Any]]:
        """
        번역 상태 확인
        
        Args:
            input_files: 확인할 입력 파일 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            Dict: 파일별 번역 상태
        """
        status = {}
        
        for input_file in input_files:
            input_path = Path(input_file)
            
            # 출력 파일 경로 추정
            if output_dir:
                output_path = Path(output_dir) / f"{input_path.stem}_korean{input_path.suffix}"
            else:
                output_path = input_path.parent / f"{input_path.stem}_korean{input_path.suffix}"
            
            file_status = {
                'input_exists': input_path.exists(),
                'output_exists': output_path.exists(),
                'input_size': input_path.stat().st_size if input_path.exists() else 0,
                'output_size': output_path.stat().st_size if output_path.exists() else 0,
                'needs_translation': not output_path.exists() or output_path.stat().st_mtime < input_path.stat().st_mtime
            }
            
            status[str(input_file)] = file_status
        
        return status
    
    def cleanup_temp_files(self, output_dir: str):
        """임시 파일 정리"""
        try:
            output_path = Path(output_dir)
            if output_path.exists():
                # raw 파일들 정리
                for raw_file in output_path.glob("*_raw.md"):
                    raw_file.unlink()
                
                # 캐시 파일 정리
                cache_file = output_path / "translation_cache.json"
                if cache_file.exists():
                    cache_file.unlink()
                    
                self.logger.info(f"임시 파일 정리 완료: {output_dir}")
        except Exception as e:
            self.logger.warning(f"임시 파일 정리 실패: {e}")