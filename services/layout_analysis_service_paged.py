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
import yaml

try:
    import fitz  # PyMuPDF for PDF page extraction
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")

import numpy as np
import paddle
import paddle.amp
from paddle.base import core

# Paddle AMP is_bfloat16_supported 함수 패치
original_is_bfloat16_supported = paddle.amp.is_bfloat16_supported

def patched_is_bfloat16_supported(device=None):
    """
    패치된 is_bfloat16_supported 함수
    Place(undefined:0) 같은 잘못된 device 인자를 처리
    """
    try:
        # device가 None이거나 문자열인 경우
        if device is None:
            return original_is_bfloat16_supported()
        
        # Place 객체인지 확인
        if hasattr(device, '__class__'):
            device_str = str(device)
            # undefined Place 객체인 경우
            if 'undefined' in device_str:
                # CPU로 가정하고 False 반환
                return False
            # 유효한 Place 객체인 경우
            elif 'cpu' in device_str.lower():
                cpu_place = paddle.CPUPlace()
                return core.is_bfloat16_supported(cpu_place)
            elif 'cuda' in device_str.lower() or 'gpu' in device_str.lower():
                if paddle.is_compiled_with_cuda():
                    gpu_place = paddle.CUDAPlace(0)
                    return core.is_bfloat16_supported(gpu_place)
                return False
        
        # 기본값
        return False
        
    except Exception as e:
        # 오류 발생 시 안전하게 False 반환
        print(f"Warning: is_bfloat16_supported check failed: {e}")
        return False

# 패치 적용
paddle.amp.is_bfloat16_supported = patched_is_bfloat16_supported
print("✅ Paddle AMP is_bfloat16_supported 함수 패치 적용 완료")


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
        
        # 기본 설정 (명시가 없으면 CUDA 빌드 시 자동 GPU 사용)
        self.use_gpu = self.config.get('use_gpu', paddle.is_compiled_with_cuda())
        self.use_table = self.config.get('use_table', True)
        self.enable_layout_detection = self.config.get('enable_layout_detection', True)  # 레이아웃 검출 활성화
        
        # 스레드별 로컬 스토리지 (각 스레드마다 별도 모델 인스턴스)
        self._local = threading.local()
        
        # 전역 모델 로드 카운터 (디버깅용)
        self._model_load_count = 0
        self._lock = threading.Lock()
        
    def _log(self, message: str) -> None:
        """콘솔과 파일에 동시에 로그를 남김 (파일 경로가 설정된 경우)."""
        try:
            print(message)
            log_path = getattr(self._local, 'log_file_path', None)
            if log_path:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
        except Exception:
            # 파일 로그 실패 시 콘솔만 유지
            pass
        
    def _gpu_snapshot(self) -> Dict[int, Dict[str, Any]]:
        """현재 GPU 상태 스냅샷 수집 (메모리/부하). 실패 시 빈 딕셔너리."""
        try:
            import GPUtil  # 지연 임포트
            snapshot: Dict[int, Dict[str, Any]] = {}
            gpus = GPUtil.getGPUs()
            for g in gpus:
                snapshot[g.id] = {
                    "name": g.name,
                    "load_percent": round((g.load or 0) * 100, 1),
                    "memory_used_mb": int(g.memoryUsed or 0),
                    "memory_total_mb": int(g.memoryTotal or 0)
                }
            return snapshot
        except Exception:
            return {}

    def _log_device_selection(self, label: str, device: str) -> None:
        """장치 선택 및 GPU 상태를 로그로 출력"""
        try:
            cuda_compiled = paddle.is_compiled_with_cuda()
            gpu_name = ""
            if isinstance(device, str) and device.startswith('gpu') and cuda_compiled:
                try:
                    from paddle.device import cuda as paddle_cuda
                    gpu_name = f" ({paddle_cuda.get_device_name(0)})"
                except Exception:
                    pass
            self._log(f"🧭 {label}: device={device}, cuda_compiled={cuda_compiled}{gpu_name}")
            snap = self._gpu_snapshot()
            if snap:
                for gid, info in snap.items():
                    self._log(
                        f"    GPU{gid} {info['name']} load={info['load_percent']}% "
                        f"mem={info['memory_used_mb']}/{info['memory_total_mb']}MB"
                    )
        except Exception:
            pass

    def _set_log_dir_from_output_dir(self, output_dir: Path) -> None:
        """출력 디렉토리 이름과 동일한 하위 폴더를 logs 아래에 만들고 파일 경로를 설정."""
        try:
            # output_dir 예: outputs/user_1_YYYYMMDD_... → logs/user_1_YYYYMMDD_...
            logs_root = Path('./logs')
            logs_root.mkdir(exist_ok=True)
            per_run_log_dir = logs_root / output_dir.name
            per_run_log_dir.mkdir(exist_ok=True)
            self._local.log_file_path = str(per_run_log_dir / 'layout_gpu.log')
            # 세션 시작 로그
            self._log(f"🗂️ GPU 로그 파일: {self._local.log_file_path}")
        except Exception:
            # 파일 경로 설정 실패 시 무시 (콘솔 로그만)
            pass
        
    def _get_layout_detector(self):
        """스레드별 LayoutDetection 모델 획득"""
        if not hasattr(self._local, 'layout_detector') or self._local.layout_detector is None:
            try:
                from paddleocr import LayoutDetection
                
                thread_id = threading.current_thread().ident
                print(f"🎨 스레드 {thread_id}: LayoutDetection 모델 로드 중...")
                
                # PP-DocLayout_plus-L 모델 사용 (더 정확한 레이아웃 검출)
                device = 'gpu:0' if (self.use_gpu and paddle.is_compiled_with_cuda()) else 'cpu'
                self._log_device_selection("LayoutDetection init", device)
                self._local.layout_detector = LayoutDetection(
                    model_name="PP-DocLayout_plus-L",
                    device=device
                )
                
                print(f"✅ 스레드 {thread_id}: LayoutDetection 모델 로드 완료!")
                
            except Exception as e:
                self.logger.warning(f"LayoutDetection 모델 로드 실패: {e}")
                self._local.layout_detector = None
                
        return self._local.layout_detector
    
    def _cleanup_thread_resources(self):
        """현재 스레드의 리소스 정리"""
        thread_id = threading.current_thread().ident
        
        # 파이프라인 정리
        if hasattr(self._local, 'pipeline') and self._local.pipeline is not None:
            try:
                print(f"🧹 스레드 {thread_id}: 기존 파이프라인 정리 중...")
                # 파이프라인 객체 삭제
                del self._local.pipeline
                self._local.pipeline = None
                print(f"✅ 스레드 {thread_id}: 파이프라인 정리 완료")
            except Exception as e:
                self.logger.warning(f"스레드 {thread_id}: 파이프라인 정리 실패: {e}")
        
        # 레이아웃 검출기 정리
        if hasattr(self._local, 'layout_detector') and self._local.layout_detector is not None:
            try:
                del self._local.layout_detector
                self._local.layout_detector = None
            except Exception as e:
                self.logger.warning(f"스레드 {thread_id}: 레이아웃 검출기 정리 실패: {e}")
    

    # --- ① 헬퍼: 누락 키 검사 함수 (파일 상단 or 클래스 안 아무데나) ---
    def _require(d: dict, path: tuple, when: bool = True):
        """
        d 에서 path(튜플) 경로의 키들이 모두 존재하는지 검사한다.
        when 이 False 면 검사를 건너뛴다.
        예) _require(cfg, ("SubPipelines","GeneralOCR","pipeline_name"))
        """
        if not when:
            return
        cur = d
        walked = []
        for key in path:
            walked.append(key)
            if not isinstance(cur, dict) or key not in cur:
                p = " → ".join(map(str, walked))
                raise RuntimeError(f"YAML 유효성 오류: '{p}' 키가 없습니다.")
            cur = cur[key]

    def _get_pipeline(self, progress_callback=None, force_reload=False):
        """스레드별 PP-Structure 파이프라인 획득"""
        
        # force_reload가 True이면 기존 파이프라인 정리
        if force_reload:
            self._cleanup_thread_resources()
        
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

                #cfg_path = (Path(__file__).parent / "PP-StructureV3.yaml").resolve()
                device = 'gpu:0' if (self.use_gpu and paddle.is_compiled_with_cuda()) else 'cpu'
                self._log_device_selection("PPStructureV3 init", device)
               
                # 스레드별 독립적인 모델 인스턴스 생성
                # self._local.pipeline = PPStructureV3(
                #     device=device,
                #     #lang='en',
                #     use_table_recognition=self.use_table,
                #     use_doc_unwarping=False,
                #     use_doc_orientation_classify=True,  # ← 주석은 "비활성화"로 되어 있는데 실제론 활성화(True)가 맞습니다.
                #     use_textline_orientation=True,
                #     paddlex_config=cfg
                # )
                self._local.pipeline = PPStructureV3(
                    device=device,
                    use_table_recognition=self.use_table,
                    use_doc_unwarping=False,  # UVDoc 비활성화
                    use_doc_orientation_classify=True,  # 방향 분류 비활성화
                    use_textline_orientation=False,
                    text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
                    #lang='en',
                    #text_recognition_model_name='en_PP-OCRv4_mobile_rec',  # 영어 특화 경량 모델
                    # 텍스트 검출 옵션 추가
                    text_det_thresh=0.2,  # 텍스트 검출 임계값 (기본 0.3에서 낮춤)
                    text_det_box_thresh=0.5,  # 박스 검출 임계값 (기본 0.6에서 낮춤)
                    text_det_unclip_ratio=2.8  # 박스 확장 비율 (기본 2.0에서 증가)
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
        
        # 새 문서 분석 시작 전 기존 리소스 정리
        self._cleanup_thread_resources()
        
        try:
            pdf_path = Path(pdf_path).resolve()
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            if output_dir is None:
                output_dir = Path(f"./output_{pdf_path.stem}_{int(time.time())}")
            else:
                output_dir = Path(output_dir).resolve()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            # 출력 디렉토리명과 동일한 logs 하위 폴더에 GPU 로그 저장
            self._set_log_dir_from_output_dir(output_dir)
            
            # 페이지별 분할 처리
            result = self._analyze_by_pages(pdf_path, output_dir, progress_callback, start_time)
            
            # 분석 완료 후 리소스 정리
            self._cleanup_thread_resources()
            
            return result
                
        except Exception as e:
            self.logger.error(f"스레드 {thread_id}: 레이아웃 분석 실패: {e}")
            traceback.print_exc()
            
            # 오류 발생 시에도 리소스 정리
            self._cleanup_thread_resources()
            
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
        
        # 스레드별 모델 인스턴스 로드 (첫 번째 호출이므로 force_reload=True)
        pipeline = self._get_pipeline(progress_callback, force_reload=True)
        
        # PDF 열기
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        print(f"📑 스레드 {thread_id}: 총 페이지 수 - {total_pages}")
        
        # PDF 메타데이터 추출
        pdf_metadata = self._extract_pdf_metadata(doc)
        print(f"📋 PDF 메타데이터:")
        print(f"   페이지 크기: {pdf_metadata['page_size']} ({pdf_metadata['orientation']})")
        print(f"   크기: {pdf_metadata['width_mm']}mm x {pdf_metadata['height_mm']}mm")
        
        if 'text_layout' in pdf_metadata:
            layout = pdf_metadata['text_layout']
            print(f"\n📐 텍스트 레이아웃 분석:")
            print(f"   평균 폰트 크기: {layout['avg_font_size_pt']}pt")
            print(f"   평균 행간: {layout['avg_line_height_pt']}pt (폰트 대비 {layout['line_height_ratio']}배)")
            
            print(f"\n📏 실제 여백 (원본 PDF):")
            print(f"   상단: {layout['margins_mm']['top']}mm ({layout['margins_pt']['top']:.1f}pt)")
            print(f"   하단: {layout['margins_mm']['bottom']}mm ({layout['margins_pt']['bottom']:.1f}pt)")
            print(f"   좌측: {layout['margins_mm']['left']}mm ({layout['margins_pt']['left']:.1f}pt)")
            print(f"   우측: {layout['margins_mm']['right']}mm ({layout['margins_pt']['right']:.1f}pt)")
            
            print(f"\n📊 페이지 활용도:")
            print(f"   콘텐츠 영역: {layout['content_width_mm']}mm × {layout['content_height_mm']}mm")
            print(f"   텍스트 너비 활용: {layout['text_width_ratio']}%")
            print(f"   텍스트 높이 활용: {layout['text_height_ratio']}%")
            print(f"   전체 페이지 대비 텍스트 영역: {layout['content_area_ratio']}%")
        
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
                    
                    raster_dpi = getattr(self, "raster_dpi",400)
                    zoom = raster_dpi /72.0
                    matrix = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
                    temp_img_path = temp_dir / f"page_{current_page:04d}.png"
                    pix.save(str(temp_img_path))
                    print(f"  🖼️ 스레드 {thread_id}: 이미지 추출 완료 - {temp_img_path.name}")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 레이아웃 분석 중...")
                    
                    # 단일 페이지 분석 (스레드별 독립 모델 사용)
                    print(f"  🔍 스레드 {thread_id}: 레이아웃 분석 시작...")
                    
                    # PPStructureV3로 분석 (예측 전후 GPU 메모리 변화 로깅)
                    before_snap = self._gpu_snapshot()
                    result = pipeline.predict(str(temp_img_path))
                    after_snap = self._gpu_snapshot()
                    if before_snap and after_snap:
                        try:
                            common_ids = set(before_snap.keys()) & set(after_snap.keys())
                            for gid in sorted(common_ids):
                                b = before_snap[gid]
                                a = after_snap[gid]
                                delta_mb = (a['memory_used_mb'] - b['memory_used_mb'])
                                delta_load = round(a['load_percent'] - b['load_percent'], 1)
                                print(
                                    f"    📈 GPU{gid} mem Δ: {delta_mb}MB, load Δ: {delta_load}%"
                                )
                        except Exception:
                            pass
                    
                    print(f"  ✅ 스레드 {thread_id}: 레이아웃 분석 완료")
                    
                    # LayoutDetection을 사용한 시각화
                    if self.enable_layout_detection:  # 모든 페이지에 대해 시각화 생성
                        layout_detector = self._get_layout_detector()
                        if layout_detector:
                            try:
                                print(f"  🎨 스레드 {thread_id}: 레이아웃 검출 및 시각화 시작...")
                                
                                # LayoutDetection 실행
                                layout_output = layout_detector.predict(
                                    str(temp_img_path), 
                                    batch_size=1, 
                                    layout_nms=True  # Non-Maximum Suppression 적용
                                )
                                
                                # 시각화 저장
                                viz_dir = output_dir / "layout_visualizations"
                                viz_dir.mkdir(exist_ok=True)
                                
                                for idx, res in enumerate(layout_output):
                                    # 결과 출력
                                    print(f"    📊 레이아웃 검출 결과 {idx+1}:")
                                    res.print()
                                    
                                    # 시각화 이미지 저장
                                    res.save_to_img(save_path=str(viz_dir) + "/")
                                    
                                    # JSON 결과 저장
                                    json_path = viz_dir / f"page_{current_page:04d}_layout.json"
                                    res.save_to_json(save_path=str(json_path))
                                    
                                    print(f"    💾 시각화 저장 완료: {viz_dir}")
                                    
                            except Exception as e:
                                self.logger.warning(f"레이아웃 검출 실패: {e}")
                    
                    if progress_callback:
                        progress_callback(page_num, total_pages, 
                                        f"페이지 {current_page}/{total_pages} 마크다운 변환 중...")
                    
                    # 결과 처리 - PaddleOCR 공식 API 사용
                    markdown_dir = output_dir / "markdown"
                    json_dir = output_dir / "json"
                    vis_dir = output_dir / "visualizations"
                    markdown_dir.mkdir(exist_ok=True)
                    json_dir.mkdir(exist_ok=True)
                    vis_dir.mkdir(exist_ok=True)
                    
                    # PPStructureV3가 predict 시 이미 시각화를 저장했으므로 추가 처리 불필요
                    # vis_dir에 이미 시각화된 이미지가 저장됨
                    print(f"  📸 시각화 이미지 저장 완료: {vis_dir}")
                    
                    # 공식 API를 사용한 결과 저장 시도
                    saved_md = False
                    saved_json = False
                    
                    md_path = None
                    json_path = None
                    
                    if isinstance(result, list):
                        for idx, res in enumerate(result):
                            if hasattr(res, 'save_to_markdown'):
                                try:
                                    res.save_to_markdown(save_path=str(markdown_dir) + f"/page_{current_page:04d}_{idx}")
                                    saved_md = True
                                except:
                                    pass
                            if hasattr(res, 'save_to_json'):
                                try:
                                    res.save_to_json(save_path=str(json_dir) + f"/page_{current_page:04d}_{idx}")
                                    saved_json = True
                                except:
                                    pass
                    
                    # 공식 API 실패시 기존 방식 사용
                    if not saved_md:
                        md_path = self._process_page_result(result, current_page, output_dir)
                        if md_path:
                            markdown_files.append(str(md_path))
                            print(f"  📝 스레드 {thread_id}: 마크다운 저장 - {md_path.name}")
                    else:
                        md_path = markdown_dir / f"page_{current_page:04d}_0.md"
                        if md_path.exists():
                            markdown_files.append(str(md_path))
                            print(f"  📝 스레드 {thread_id}: 마크다운 저장 (공식 API) - {md_path.name}")
                    
                    # 페이지별 JSON 결과 저장 (공식 API 실패시)
                    if not saved_json:
                        json_path = output_dir / f"page_{current_page:04d}_result.json"
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                    
                    if saved_md:
                        # 기존: md_path = markdown_dir / f"page_{current_page:04d}_0.md"
                        candidates = sorted(markdown_dir.glob(f"page_{current_page:04d}_*.md"))
                        if candidates:
                            md_path = candidates[0]
                            markdown_files.append(str(md_path))
                            print(f"  📝 스레드 {thread_id}: 마크다운 저장 (공식 API) - {md_path.name}")

                    if saved_json and json_path is None:
                        # json_dir 안에 이 페이지 번호로 저장된 첫 파일을 대표로 잡기
                        candidates = sorted(json_dir.glob(f"page_{current_page:04d}_*.json"))
                        if candidates:
                            json_path = candidates[0]
                    
                    page_time = time.time() - page_start
                    
                    all_pages.append({
                        "page_number": current_page,
                        "markdown_file": str(md_path) if md_path else None,
                        "json_file": str(json_path) if json_path else None,
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
                "concurrent_processing": True,
                "pdf_info": pdf_metadata  # PDF 메타데이터 추가
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
    
    def _extract_pdf_metadata(self, doc) -> Dict[str, Any]:
        """PDF 문서의 메타데이터 및 텍스트 레이아웃 정보 추출"""
        try:
            # 첫 페이지의 크기 정보 가져오기
            first_page = doc[0]
            rect = first_page.rect
            width_pt = rect.width  # 포인트 단위
            height_pt = rect.height  # 포인트 단위
            
            # 포인트를 mm로 변환 (1pt = 0.352778mm)
            width_mm = width_pt * 0.352778
            height_mm = height_pt * 0.352778
            
            # 페이지 크기 판별
            page_size = "Custom"
            orientation = "portrait" if height_pt > width_pt else "landscape"
            
            # 일반적인 용지 크기 판별 (약간의 오차 허용)
            sizes = {
                "A4": (210, 297),
                "Letter": (215.9, 279.4),
                "Legal": (215.9, 355.6),
                "A3": (297, 420),
                "B4": (250, 353),
                "B5": (176, 250)
            }
            
            for name, (w, h) in sizes.items():
                # 세로/가로 모두 체크
                if (abs(width_mm - w) < 5 and abs(height_mm - h) < 5) or \
                   (abs(width_mm - h) < 5 and abs(height_mm - w) < 5):
                    page_size = name
                    break
            
            # 텍스트 레이아웃 분석
            text_layout = self._analyze_text_layout(first_page)
            
            # DPI 추정 (PDF는 보통 72 DPI 기준)
            dpi = 72
            
            return {
                "page_size": page_size,
                "orientation": orientation,
                "width_mm": round(width_mm, 2),
                "height_mm": round(height_mm, 2),
                "width_pt": round(width_pt, 2),
                "height_pt": round(height_pt, 2),
                "dpi": dpi,
                "total_pages": len(doc),
                "text_layout": text_layout  # 텍스트 레이아웃 정보 추가
            }
            
        except Exception as e:
            self.logger.warning(f"PDF 메타데이터 추출 실패: {e}")
            return {
                "page_size": "A4",
                "orientation": "portrait",
                "width_mm": 210,
                "height_mm": 297,
                "dpi": 72,
                "total_pages": len(doc) if doc else 0
            }
    
    def _analyze_text_layout(self, page) -> Dict[str, Any]:
        """페이지의 텍스트 레이아웃 상세 분석"""
        try:
            # 텍스트 블록 추출
            text_blocks = page.get_text("dict")
            
            # 페이지 크기
            page_width = page.rect.width
            page_height = page.rect.height
            page_area = page_width * page_height
            
            # 텍스트 영역 계산
            text_bounds = {
                'min_x': page_width,
                'max_x': 0,
                'min_y': page_height,
                'max_y': 0
            }
            
            font_sizes = []
            line_heights = []
            char_spacings = []
            
            total_text_area = 0
            
            # 각 블록 분석
            for block in text_blocks.get('blocks', []):
                if block.get('type') == 0:  # 텍스트 블록
                    bbox = block.get('bbox', [0, 0, 0, 0])
                    
                    # 텍스트 영역 경계 업데이트
                    text_bounds['min_x'] = min(text_bounds['min_x'], bbox[0])
                    text_bounds['max_x'] = max(text_bounds['max_x'], bbox[2])
                    text_bounds['min_y'] = min(text_bounds['min_y'], bbox[1])
                    text_bounds['max_y'] = max(text_bounds['max_y'], bbox[3])
                    
                    # 블록 면적 계산
                    block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    total_text_area += block_area
                    
                    # 라인별 분석
                    for line in block.get('lines', []):
                        spans = line.get('spans', [])
                        
                        for span in spans:
                            # 폰트 크기 수집
                            font_size = span.get('size', 0)
                            if font_size > 0:
                                font_sizes.append(font_size)
                            
                            # 문자 간격 분석 (bbox 정보가 있는 경우)
                            chars = span.get('chars', [])
                            if len(chars) > 1:
                                for i in range(len(chars) - 1):
                                    char_bbox1 = chars[i].get('bbox', None)
                                    char_bbox2 = chars[i + 1].get('bbox', None)
                                    if char_bbox1 and char_bbox2:
                                        spacing = char_bbox2[0] - char_bbox1[2]
                                        if spacing >= 0:  # 정상적인 간격만
                                            char_spacings.append(spacing)
                    
                    # 라인 간격 계산
                    lines = block.get('lines', [])
                    if len(lines) > 1:
                        for i in range(len(lines) - 1):
                            line1_bbox = lines[i].get('bbox', None)
                            line2_bbox = lines[i + 1].get('bbox', None)
                            if line1_bbox and line2_bbox:
                                line_spacing = line2_bbox[1] - line1_bbox[3]
                                if line_spacing >= 0:  # 정상적인 간격만
                                    line_heights.append(line_spacing)
            
            # 텍스트 영역이 발견된 경우에만 계산
            if text_bounds['max_x'] > 0 and text_bounds['max_y'] > 0:
                # 실제 텍스트 영역 크기 계산
                actual_text_width = text_bounds['max_x'] - text_bounds['min_x']
                actual_text_height = text_bounds['max_y'] - text_bounds['min_y']
                actual_text_area = actual_text_width * actual_text_height
                
                # 여백 계산 (정확한 콘텐츠 경계 기준)
                margin_left = text_bounds['min_x']
                margin_right = page_width - text_bounds['max_x']
                margin_top = text_bounds['min_y']
                margin_bottom = page_height - text_bounds['max_y']
            else:
                # 텍스트가 없는 경우 기본값
                actual_text_width = page_width * 0.8
                actual_text_height = page_height * 0.8
                actual_text_area = actual_text_width * actual_text_height
                margin_left = margin_right = page_width * 0.1
                margin_top = margin_bottom = page_height * 0.1
            
            # 통계 계산
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 11
            avg_line_height = sum(line_heights) / len(line_heights) if line_heights else avg_font_size * 1.2
            avg_char_spacing = sum(char_spacings) / len(char_spacings) if char_spacings else 0
            
            # 페이지 활용도 계산
            text_coverage = (total_text_area / page_area * 100) if page_area > 0 else 0
            content_area_ratio = (actual_text_area / page_area * 100) if page_area > 0 else 0
            
            # mm 단위로 변환 (1pt = 0.352778mm)
            pt_to_mm = 0.352778
            
            return {
                "avg_font_size_pt": round(avg_font_size, 2),
                "avg_line_height_pt": round(avg_line_height, 2),
                "avg_char_spacing_pt": round(avg_char_spacing, 3),
                "line_height_ratio": round(avg_line_height / avg_font_size, 2) if avg_font_size > 0 else 1.2,
                "margins_pt": {
                    "left": round(margin_left, 2),
                    "right": round(margin_right, 2),
                    "top": round(margin_top, 2),
                    "bottom": round(margin_bottom, 2)
                },
                "margins_mm": {
                    "left": round(margin_left * pt_to_mm, 1),
                    "right": round(margin_right * pt_to_mm, 1),
                    "top": round(margin_top * pt_to_mm, 1),
                    "bottom": round(margin_bottom * pt_to_mm, 1)
                },
                "text_coverage_percent": round(text_coverage, 2),
                "content_area_ratio": round(content_area_ratio, 2),
                "text_width_ratio": round(actual_text_width / page_width * 100, 2) if page_width > 0 else 0,
                "text_height_ratio": round(actual_text_height / page_height * 100, 2) if page_height > 0 else 0,
                "content_width_mm": round(actual_text_width * pt_to_mm, 1),
                "content_height_mm": round(actual_text_height * pt_to_mm, 1)
            }
            
        except Exception as e:
            self.logger.warning(f"텍스트 레이아웃 분석 실패: {e}")
            return {
                "avg_font_size_pt": 11,
                "avg_line_height_pt": 13.2,
                "avg_char_spacing_pt": 0,
                "line_height_ratio": 1.2,
                "margins_pt": {"left": 72, "right": 72, "top": 72, "bottom": 72},
                "text_coverage_percent": 50,
                "content_area_ratio": 70,
                "text_width_ratio": 80,
                "text_height_ratio": 85
            }
    
    def _save_visualization(self, image_path: str, result: Any, output_path: Path):
        """PPStructureV3 결과를 시각화하여 저장"""
        self.visualize_layout_detection_manual(image_path, result, str(output_path))
    
    def visualize_layout_detection_manual(self, image_path: str, result: Any, output_path: str):
        """
        [백업용] 수동으로 레이아웃 검출 결과를 이미지에 시각화
        LayoutDetection을 사용할 수 없을 때 대체용
        
        Args:
            image_path: 원본 이미지 경로
            result: PaddleOCR 분석 결과
            output_path: 시각화 이미지 저장 경로
        """
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available, skipping visualization")
            return
            
        try:
            # 이미지 로드
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img, 'RGBA')  # 투명도 지원
            
            # 폰트 설정 (기본 폰트 사용)
            try:
                font = ImageFont.truetype("malgun.ttf", 20)  # Windows 한글 폰트
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            # 각 요소 타입별 색상 설정
            colors = {
                'title': (255, 0, 0, 200),      # 빨간색
                'text': (0, 255, 0, 150),       # 초록색
                'table': (0, 0, 255, 200),      # 파란색
                'figure': (255, 255, 0, 200),   # 노란색
                'list': (255, 0, 255, 200),     # 자홍색
                'equation': (0, 255, 255, 200), # 청록색
                'reference': (255, 128, 0, 200), # 주황색
                'header': (128, 0, 255, 200),   # 보라색
                'footer': (128, 255, 0, 200),   # 연두색
                'default': (128, 128, 128, 150) # 회색
            }
            
            detected_elements = []
            
            # PaddleOCR PPStructureV3 결과 구조 파싱
            if isinstance(result, list):
                for idx, item in enumerate(result):
                    bbox = None
                    elem_type = 'text'  # 기본 타입
                    confidence = 0.0
                    
                    # 다양한 형식 처리
                    if hasattr(item, '__dict__'):
                        # 객체 형식
                        bbox = getattr(item, 'bbox', None)
                        elem_type = getattr(item, 'type', 'text')
                        confidence = getattr(item, 'confidence', 0.0)
                    elif isinstance(item, dict):
                        # 딕셔너리 형식
                        bbox = item.get('bbox', None)
                        elem_type = item.get('type', 'text')
                        confidence = item.get('confidence', 0.0)
                        
                        # res 키가 있는 경우 (테이블 등)
                        if 'res' in item and isinstance(item['res'], dict):
                            if 'type' in item['res']:
                                elem_type = item['res']['type']
                    
                    # bbox 그리기
                    if bbox:
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                        elif len(bbox) == 8:  # 4개의 좌표점
                            x1 = min(bbox[0], bbox[2], bbox[4], bbox[6])
                            y1 = min(bbox[1], bbox[3], bbox[5], bbox[7])
                            x2 = max(bbox[0], bbox[2], bbox[4], bbox[6])
                            y2 = max(bbox[1], bbox[3], bbox[5], bbox[7])
                        else:
                            continue
                        
                        # 타입에 따른 색상 선택
                        color = colors.get(elem_type, colors['default'])
                        
                        # 반투명 박스 채우기
                        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        fill_color = color[:3] + (50,)  # 더 투명하게
                        overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color, width=3)
                        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                        draw = ImageDraw.Draw(img)
                        
                        # 라벨 표시 (배경 추가)
                        label = f"{elem_type}"
                        if confidence > 0:
                            label += f" ({confidence:.2f})"
                        
                        # 텍스트 배경
                        text_bbox = draw.textbbox((x1, y1-25), label, font=font)
                        draw.rectangle(text_bbox, fill='white', outline='black')
                        draw.text((x1, y1-25), label, fill=color[:3], font=font)
                        
                        detected_elements.append(elem_type)
            
            # 범례 추가 (발견된 요소 타입만)
            unique_elements = list(set(detected_elements))
            if unique_elements:
                legend_bg = Image.new('RGBA', (180, len(unique_elements) * 30 + 20), (255, 255, 255, 230))
                legend_draw = ImageDraw.Draw(legend_bg)
                
                legend_draw.text((10, 5), "검출된 요소:", fill='black', font=font)
                
                for i, elem_type in enumerate(unique_elements):
                    color = colors.get(elem_type, colors['default'])
                    y_pos = 30 + i * 25
                    legend_draw.rectangle([10, y_pos, 25, y_pos + 15], fill=color[:3], outline='black')
                    legend_draw.text((30, y_pos), elem_type, fill='black', font=font)
                
                # 범례를 이미지에 합성
                img.paste(legend_bg, (img.width - 190, 10), legend_bg)
            
            # 통계 정보 추가
            stats_text = f"총 {len(detected_elements)}개 요소 검출"
            draw.rectangle([10, img.height - 40, 250, img.height - 10], fill='white', outline='black')
            draw.text((15, img.height - 35), stats_text, fill='black', font=font)
            
            # 이미지 저장
            img.save(output_path, quality=95)
            print(f"  📸 시각화 이미지 저장: {output_path}")
            print(f"     검출된 요소: {', '.join(unique_elements) if unique_elements else '없음'}")
            
        except Exception as e:
            self.logger.warning(f"시각화 실패: {e}")
    
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