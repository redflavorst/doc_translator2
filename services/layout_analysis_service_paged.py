# services/layout_analysis_service_paged.py (동시성 문제 해결)
"""
페이지별 진행률 추적이 가능한 레이아웃 분석 서비스 (동시 사용자 지원)
각 스레드마다 별도의 PPStructureV3 모델 인스턴스를 사용하여 동시성 문제 해결
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import traceback
import tempfile
import shutil
import threading

try:
    import fitz  # PyMuPDF for PDF page extraction
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF")


@dataclass
class LayoutAnalysisResult:
    """레이아웃 분석 결과"""
    success: bool
    pages: List[Dict[str, Any]]
    markdown_files: List[str]
    confidence: float
    processing_time: float
    output_dir: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LayoutAnalysisServicePaged:
    """
    페이지별 진행률 추적이 가능한 레이아웃 분석 서비스 (멀티 스레드 안전)
    각 스레드마다 독립적인 PaddleOCR 인스턴스를 사용
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 기본 설정
        self.use_gpu = self.config.get('use_gpu', False)
        self.use_table = self.config.get('use_table', True)
        
        # 스레드별 로컬 스토리지 (각 스레드마다 별도 모델 인스턴스)
        self._local = threading.local()
        
        # 전역 모델 로드 카운터 (디버깅용)
        self._model_load_count = 0
        self._lock = threading.Lock()
        
    def _get_pipeline(self, progress_callback=None):
        """스레드별 PP-Structure 파이프라인 획득"""
        
        # 현재 스레드에 파이프라인이 없는 경우에만 생성
        if not hasattr(self._local, 'pipeline') or self._local.pipeline is None:
            try:
                from paddleocr import PPStructureV3
                
                # 스레드 정보
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name
                
                with self._lock:
                    self._model_load_count += 1
                    current_count = self._model_load_count
                
                print(f"\n{'='*60}")
                print(f"🔄 스레드별 PPStructureV3 모델 로드")
                print(f"   스레드 ID: {thread_id}")
                print(f"   스레드 이름: {thread_name}")
                print(f"   로드 순서: {current_count}번째")
                print(f"{'='*60}")
                
                if progress_callback:
                    progress_callback(0, 100, f"분석 모델 로드 중... (스레드 {current_count})")
                
                start_load = time.time()
                
                # 스레드별 독립적인 모델 인스턴스 생성
                self._local.pipeline = PPStructureV3(
                    device='cpu',  # CPU 사용으로 안정성 확보
                    use_table_recognition=self.use_table,
                    use_doc_unwarping=False,  # UVDoc 비활성화
                    use_doc_orientation_classify=False  # 방향 분류 비활성화
                )
                
                load_time = time.time() - start_load
                
                print(f"✅ 스레드 {thread_id}: 모델 로드 완료!")
                print(f"   소요시간: {load_time:.2f}초")
                print(f"   메모리 사용: 독립 인스턴스")
                print(f"{'='*60}\n")
                
                if progress_callback:
                    progress_callback(100, 100, "분석 모델 로드 완료")
                        
            except Exception as e:
                print(f"❌ 스레드 {thread_id}: 모델 로드 실패: {e}")
                raise RuntimeError(f"PPStructureV3 초기화 실패: {e}")
                
        return self._local.pipeline
    
    def analyze_document(self, pdf_path: str, output_dir: str = None, 
                        progress_callback=None) -> LayoutAnalysisResult:
        """
        PDF 문서를 페이지별로 분석 (스레드 안전)
        
        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리
            progress_callback: 진행률 콜백 (current, total, message)
        """
        if not PYMUPDF_AVAILABLE:
            return LayoutAnalysisResult(
                success=False,
                pages=[],
                markdown_files=[],
                confidence=0.0,
                processing_time=0,
                output_dir="",
                error="PyMuPDF가 설치되지 않았습니다. pip install PyMuPDF를 실행하세요."
            )
        
        start_time = time.time()
        thread_id = threading.current_thread().ident
        
        print(f"\n📄 스레드 {thread_id}: PDF 분석 시작")
        
        try:
            pdf_path = Path(pdf_path).resolve()
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            if output_dir is None:
                output_dir = Path(f"./output_{pdf_path.stem}_{int(time.time())}")
            else:
                output_dir = Path(output_dir).resolve()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 페이지별 분할 처리
            return self._analyze_by_pages(pdf_path, output_dir, progress_callback, start_time)
                
        except Exception as e:
            self.logger.error(f"스레드 {thread_id}: 레이아웃 분석 실패: {e}")
            traceback.print_exc()
            return LayoutAnalysisResult(
                success=False,
                pages=[],
                markdown_files=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                output_dir=str(output_dir) if 'output_dir' in locals() else "",
                error=str(e)
            )
    
    def _analyze_by_pages(self, pdf_path: Path, output_dir: Path, 
                         progress_callback, start_time: float) -> LayoutAnalysisResult:
        """페이지별로 분할하여 처리 (스레드별 독립 실행)"""
        
        thread_id = threading.current_thread().ident
        print(f"📊 스레드 {thread_id}: 페이지별 분석 시작 - {pdf_path.name}")
        
        # 스레드별 모델 인스턴스 로드
        pipeline = self._get_pipeline(progress_callback)
        
        # PDF 열기
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        print(f"📑 스레드 {thread_id}: 총 페이지 수 - {total_pages}")
        
        if progress_callback:
            progress_callback(0, total_pages, f"PDF 분석 준비 중... (총 {total_pages}페이지)")
        
        all_pages = []
        markdown_files = []
        
        # 임시 디렉토리 생성 (스레드별로 독립)
        temp_dir = output_dir / f"temp_images_{thread_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 각 페이지를 개별 처리
            for page_num in range(total_pages):
                try:
                    page_start = time.time()
                    current_page = page_num + 1
                    
                    print(f"📄 스레드 {thread_id}: 페이지 {current_page}/{total_pages} 처리 시작")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 추출 중...")
                    
                    # 페이지를 이미지로 추출 (고해상도)
                    page = doc[page_num]
                    matrix = fitz.Matrix(2, 2)  # 2배 해상도
                    pix = page.get_pixmap(matrix=matrix)
                    temp_img_path = temp_dir / f"page_{current_page:04d}.png"
                    pix.save(str(temp_img_path))
                    print(f"  🖼️ 스레드 {thread_id}: 이미지 추출 완료 - {temp_img_path.name}")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 레이아웃 분석 중...")
                    
                    # 단일 페이지 분석 (스레드별 독립 모델 사용)
                    print(f"  🔍 스레드 {thread_id}: 레이아웃 분석 시작...")
                    result = pipeline.predict(input=str(temp_img_path))
                    print(f"  ✅ 스레드 {thread_id}: 레이아웃 분석 완료")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 마크다운 변환 중...")
                    
                    # 결과 처리 및 마크다운 생성
                    md_path = self._process_page_result(result, current_page, output_dir)
                    if md_path:
                        markdown_files.append(str(md_path))
                        print(f"  📝 스레드 {thread_id}: 마크다운 저장 - {md_path.name}")
                    
                    # 페이지별 JSON 결과 저장
                    json_path = output_dir / f"page_{current_page:04d}_result.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                    
                    page_time = time.time() - page_start
                    
                    all_pages.append({
                        "page_number": current_page,
                        "markdown_file": str(md_path) if md_path else None,
                        "json_file": str(json_path),
                        "processing_time": page_time,
                        "thread_id": thread_id
                    })
                    
                    print(f"  ⏱️ 스레드 {thread_id}: 페이지 처리 완료 (소요시간: {page_time:.2f}초)")
                    
                    if progress_callback:
                        progress_callback(current_page, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 완료")
                    
                except Exception as e:
                    print(f"  ❌ 스레드 {thread_id}: 페이지 {current_page} 처리 실패: {e}")
                    self.logger.warning(f"스레드 {thread_id}: 페이지 {current_page} 처리 실패: {e}")
                    all_pages.append({
                        "page_number": current_page,
                        "error": str(e),
                        "thread_id": thread_id
                    })
            
            # 임시 디렉토리 정리
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        finally:
            doc.close()
        
        total_time = time.time() - start_time
        print(f"\n✅ 스레드 {thread_id}: 전체 분석 완료!")
        print(f"   총 소요시간: {total_time:.2f}초")
        print(f"   페이지당 평균: {total_time/total_pages:.2f}초")
        print(f"   생성된 마크다운 파일: {len(markdown_files)}개")
        print(f"   스레드 독립성: 확보됨")
        
        return LayoutAnalysisResult(
            success=True,
            pages=all_pages,
            markdown_files=markdown_files,
            confidence=0.95,
            processing_time=total_time,
            output_dir=str(output_dir),
            metadata={
                "total_pages": total_pages,
                "method": "page_by_page_threaded",
                "thread_id": thread_id,
                "avg_page_time": total_time / total_pages if total_pages > 0 else 0,
                "concurrent_processing": True
            }
        )
    
    def _process_page_result(self, result, page_num: int, output_dir: Path) -> Optional[Path]:
        """단일 페이지 결과를 마크다운으로 변환 (스레드 안전)"""
        try:
            md_text = ""
            
            # 결과에서 마크다운 텍스트 추출 (오직 'markdown_texts' 값만 저장)
            if isinstance(result, list):
                for item in result:
                    if hasattr(item, 'markdown'):
                        md_info = getattr(item, 'markdown')
                        if isinstance(md_info, dict):
                            txt = md_info.get('markdown_texts') or md_info.get('markdown') or ''
                            md_text += str(txt or '') + "\n"
                        else:
                            md_text += str(md_info or '') + "\n"
                    elif isinstance(item, dict):
                        md_info = item.get('markdown')
                        if isinstance(md_info, dict):
                            txt = md_info.get('markdown_texts') or md_info.get('markdown') or ''
                            md_text += str(txt or '') + "\n"
                        elif 'text' in item:
                            md_text += str(item.get('text') or '') + "\n"
            elif hasattr(result, 'markdown'):
                md_info = getattr(result, 'markdown')
                if isinstance(md_info, dict):
                    md_text = str(md_info.get('markdown_texts') or md_info.get('markdown') or '')
                else:
                    md_text = str(md_info or '')
            elif isinstance(result, dict):
                md_info = result.get('markdown')
                if isinstance(md_info, dict):
                    md_text = str(md_info.get('markdown_texts') or md_info.get('markdown') or '')
                else:
                    md_text = ''
            else:
                md_text = ''
            
            if not md_text.strip():
                md_text = f"# Page {page_num}\n\n[No text content detected]"
            
            # 마크다운 파일 저장 (스레드 안전)
            md_path = output_dir / f"page_{page_num:04d}.md"
            md_path.write_text(md_text, encoding='utf-8')
            
            return md_path
            
        except Exception as e:
            thread_id = threading.current_thread().ident
            self.logger.error(f"스레드 {thread_id}: 페이지 {page_num} 마크다운 변환 실패: {e}")
            traceback.print_exc()
            return None
    
    def get_thread_info(self) -> Dict[str, Any]:
        """현재 스레드의 모델 정보 반환 (디버깅용)"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        has_pipeline = hasattr(self._local, 'pipeline') and self._local.pipeline is not None
        
        return {
            "thread_id": thread_id,
            "thread_name": thread_name,
            "has_pipeline": has_pipeline,
            "total_models_loaded": self._model_load_count
        }


def test_concurrent_analysis():
    """동시성 테스트 함수"""
    import sys
    import threading
    import time
    
    def progress_callback(current, total, message):
        """진행률 표시 콜백"""
        thread_id = threading.current_thread().ident
        if total > 0:
            percent = (current / total) * 100
            print(f"스레드 {thread_id}: [{percent:.1f}%] {message}")
    
    def analyze_worker(service, pdf_path, worker_id):
        """워커 스레드 함수"""
        print(f"\n🚀 워커 {worker_id} 시작")
        
        result = service.analyze_document(
            pdf_path,
            output_dir=f"./test_output_worker_{worker_id}_{int(time.time())}",
            progress_callback=progress_callback
        )
        
        if result.success:
            print(f"✅ 워커 {worker_id} 성공!")
            print(f"   처리된 페이지: {len(result.pages)}")
            print(f"   생성된 마크다운: {len(result.markdown_files)}")
            print(f"   처리 시간: {result.processing_time:.2f}초")
            print(f"   스레드 정보: {result.metadata.get('thread_id')}")
        else:
            print(f"❌ 워커 {worker_id} 실패: {result.error}")
    
    # 테스트 실행
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("PDF 파일 경로를 입력하세요: ")
    
    if not Path(pdf_path).exists():
        print(f"❌ 파일이 존재하지 않습니다: {pdf_path}")
        return
    
    service = LayoutAnalysisServicePaged()
    
    # 동시에 2개 워커 실행
    print("🔄 동시성 테스트: 2개 워커 동시 실행")
    
    threads = []
    for i in range(2):
        thread = threading.Thread(
            target=analyze_worker,
            args=(service, pdf_path, i+1),
            name=f"Worker-{i+1}"
        )
        threads.append(thread)
        thread.start()
        time.sleep(1)  # 1초 간격으로 시작
    
    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()
    
    print("\n🎯 동시성 테스트 완료!")
    print(f"총 로드된 모델 수: {service._model_load_count}")


if __name__ == "__main__":
    test_concurrent_analysis()