# services/layout_analysis_service_paged.py
"""
페이지별 진행률 추적이 가능한 레이아웃 분석 서비스 (테스트용)
PPStructureV3 모델은 한 번만 로드하고 각 페이지마다 재사용
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
    """페이지별 진행률 추적이 가능한 레이아웃 분석 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 기본 설정
        self.use_gpu = self.config.get('use_gpu', False)
        self.use_table = self.config.get('use_table', True)
        
        # PP-Structure 파이프라인 (싱글톤)
        self._pipeline = None
        
    def _get_pipeline(self, progress_callback=None):
        """PP-Structure 파이프라인 초기화 (한 번만 로드)"""
        if self._pipeline is None:
            try:
                from paddleocr import PPStructureV3
                
                # 모델 로딩 진행 상황 알림
                if progress_callback:
                    progress_callback(0, 100, "분석 모델 로드 중... (최초 1회)")
                
                # 모델은 최초 1회만 로드
                print("="*50)
                print("PPStructureV3 모델 로드 시작... (최초 1회만)")
                start_load = time.time()
                
                self._pipeline = PPStructureV3(
                    device='cpu',
                    use_table_recognition=self.use_table,
                    use_doc_unwarping=False,  # UVDoc 비활성화
                    use_doc_orientation_classify=False
                )
                
                load_time = time.time() - start_load
                print(f"PPStructureV3 모델 로드 완료! (소요시간: {load_time:.2f}초)")
                print("="*50)
                
                if progress_callback:
                    progress_callback(100, 100, "분석 모델 로드 완료")
                
            except Exception as e:
                raise RuntimeError(f"PPStructureV3 초기화 실패: {e}")
                
        return self._pipeline
    
    def analyze_document(self, pdf_path: str, output_dir: str = None, 
                        progress_callback=None) -> LayoutAnalysisResult:
        """
        PDF 문서를 페이지별로 분석 (실시간 진행률 추적)
        
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
            self.logger.error(f"레이아웃 분석 실패: {e}")
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
        """페이지별로 분할하여 처리 (실시간 진행률 추적)"""
        
        print(f"\n📄 PDF 페이지별 분석 시작: {pdf_path.name}")
        
        # 모델 로드 (한 번만!) - 진행 상황 알림
        pipeline = self._get_pipeline(progress_callback)
        
        # PDF 열기
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        print(f"총 페이지 수: {total_pages}")
        
        if progress_callback:
            progress_callback(0, total_pages, f"PDF 분석 준비 중... (총 {total_pages}페이지)")
        
        all_pages = []
        markdown_files = []
        
        # 임시 디렉토리 생성
        temp_dir = output_dir / "temp_images"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 각 페이지를 개별 처리
            for page_num in range(total_pages):
                try:
                    page_start = time.time()
                    current_page = page_num + 1
                    
                    print(f"\n--- 페이지 {current_page}/{total_pages} 처리 시작 ---")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 추출 중...")
                    
                    # 페이지를 이미지로 추출 (고해상도)
                    page = doc[page_num]
                    matrix = fitz.Matrix(2, 2)  # 2배 해상도
                    pix = page.get_pixmap(matrix=matrix)
                    temp_img_path = temp_dir / f"page_{current_page:04d}.png"
                    pix.save(str(temp_img_path))
                    print(f"  이미지 추출 완료: {temp_img_path.name}")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 레이아웃 분석 중...")
                    
                    # 단일 페이지 분석 (이미 로드된 모델 사용)
                    print(f"  레이아웃 분석 시작...")
                    result = pipeline.predict(input=str(temp_img_path))
                    print(f"  레이아웃 분석 완료")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 마크다운 변환 중...")
                    
                    # 결과 처리 및 마크다운 생성
                    md_path = self._process_page_result(result, current_page, output_dir)
                    if md_path:
                        markdown_files.append(str(md_path))
                        print(f"  마크다운 저장: {md_path.name}")
                    
                    # 페이지별 JSON 결과 저장
                    json_path = output_dir / f"page_{current_page:04d}_result.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                    
                    page_time = time.time() - page_start
                    
                    all_pages.append({
                        "page_number": current_page,
                        "markdown_file": str(md_path) if md_path else None,
                        "json_file": str(json_path),
                        "processing_time": page_time
                    })
                    
                    print(f"  페이지 처리 완료 (소요시간: {page_time:.2f}초)")
                    
                    if progress_callback:
                        progress_callback(current_page, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 완료")
                    
                except Exception as e:
                    print(f"  ❌ 페이지 {current_page} 처리 실패: {e}")
                    self.logger.warning(f"페이지 {current_page} 처리 실패: {e}")
                    all_pages.append({
                        "page_number": current_page,
                        "error": str(e)
                    })
            
            # 임시 디렉토리 정리
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        finally:
            doc.close()
        
        total_time = time.time() - start_time
        print(f"\n✅ 전체 분석 완료!")
        print(f"총 소요시간: {total_time:.2f}초")
        print(f"페이지당 평균: {total_time/total_pages:.2f}초")
        print(f"생성된 마크다운 파일: {len(markdown_files)}개")
        
        return LayoutAnalysisResult(
            success=True,
            pages=all_pages,
            markdown_files=markdown_files,
            confidence=0.95,
            processing_time=total_time,
            output_dir=str(output_dir),
            metadata={
                "total_pages": total_pages,
                "method": "page_by_page",
                "avg_page_time": total_time / total_pages if total_pages > 0 else 0
            }
        )
    
    def _process_page_result(self, result, page_num: int, output_dir: Path) -> Optional[Path]:
        """단일 페이지 결과를 마크다운으로 변환"""
        try:
            md_text = ""
            
            # 결과에서 마크다운 텍스트 추출
            if isinstance(result, list):
                for item in result:
                    if hasattr(item, 'markdown'):
                        md_text += str(item.markdown) + "\n"
                    elif isinstance(item, dict):
                        if 'markdown' in item:
                            md_text += str(item['markdown']) + "\n"
                        if 'text' in item:
                            md_text += str(item['text']) + "\n"
            elif hasattr(result, 'markdown'):
                md_text = str(result.markdown)
            else:
                # 결과를 문자열로 변환
                md_text = str(result)
            
            if not md_text.strip():
                md_text = f"# Page {page_num}\n\n[No text content detected]"
            
            # 마크다운 파일 저장
            md_path = output_dir / f"page_{page_num:04d}.md"
            md_path.write_text(md_text, encoding='utf-8')
            
            return md_path
            
        except Exception as e:
            self.logger.error(f"페이지 {page_num} 마크다운 변환 실패: {e}")
            traceback.print_exc()
            return None


def test_paged_analysis():
    """테스트 함수"""
    import sys
    
    def progress_callback(current, total, message):
        """진행률 표시 콜백"""
        if total > 0:
            percent = (current / total) * 100
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r[{bar}] {percent:.1f}% - {message}", end='', flush=True)
            if current == total:
                print()  # 완료 시 줄바꿈
    
    # 테스트 실행
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("PDF 파일 경로를 입력하세요: ")
    
    service = LayoutAnalysisServicePaged()
    result = service.analyze_document(
        pdf_path,
        output_dir=f"./test_output_paged_{int(time.time())}",
        progress_callback=progress_callback
    )
    
    if result.success:
        print(f"\n✅ 분석 성공!")
        print(f"처리된 페이지: {len(result.pages)}")
        print(f"생성된 마크다운: {len(result.markdown_files)}")
        print(f"출력 디렉토리: {result.output_dir}")
    else:
        print(f"\n❌ 분석 실패: {result.error}")


if __name__ == "__main__":
    test_paged_analysis()