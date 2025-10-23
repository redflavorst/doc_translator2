# services/deepseek_ocr_service.py
"""
DeepSeek-OCR 서비스
PDF 이미지에서 텍스트와 레이아웃 정보 추출 (bbox 좌표 포함)
"""

import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import traceback

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")


@dataclass
class OCRResult:
    """OCR 결과"""
    success: bool
    components: List[Dict[str, Any]]
    page_size: Dict[str, int]
    confidence: float
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DeepSeekOCRService:
    """
    DeepSeek-OCR 서비스
    이미지에서 텍스트 + Bounding Box 좌표 추출
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # GPU 사용 여부
        self.use_gpu = self.config.get('use_gpu', torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False)
        self.model_name = self.config.get('model_name', 'deepseek-ai/DeepSeek-OCR')

        # 스레드별 로컬 스토리지 (각 스레드마다 별도 모델 인스턴스)
        self._local = threading.local()
        self._model_load_count = 0
        self._lock = threading.Lock()

    def _get_model_and_tokenizer(self):
        """스레드별 DeepSeek-OCR 모델 및 토크나이저 획득"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers 패키지가 설치되지 않았습니다.")

        if not hasattr(self._local, 'model') or self._local.model is None:
            thread_id = threading.current_thread().ident

            with self._lock:
                self._model_load_count += 1
                current_count = self._model_load_count

            print(f"\n{'='*60}")
            print(f"🔄 DeepSeek-OCR 모델 로드")
            print(f"   스레드 ID: {thread_id}")
            print(f"   로드 순서: {current_count}번째")
            print(f"{'='*60}")

            start_load = time.time()

            try:
                # 토크나이저 로드
                self._local.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                # 모델 로드
                device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'

                if device == 'cuda':
                    self._local.model = AutoModel.from_pretrained(
                        self.model_name,
                        _attn_implementation='flash_attention_2',
                        trust_remote_code=True,
                        use_safetensors=True
                    ).eval().cuda().to(torch.bfloat16)
                else:
                    self._local.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        use_safetensors=True
                    ).eval()

                load_time = time.time() - start_load

                print(f"✅ 스레드 {thread_id}: DeepSeek-OCR 모델 로드 완료!")
                print(f"   소요시간: {load_time:.2f}초")
                print(f"   Device: {device}")
                print(f"{'='*60}\n")

            except Exception as e:
                print(f"❌ 스레드 {thread_id}: 모델 로드 실패: {e}")
                raise RuntimeError(f"DeepSeek-OCR 초기화 실패: {e}")

        return self._local.model, self._local.tokenizer

    def extract_layout(self, image_path: str, output_dir: Optional[str] = None,
                      progress_callback=None) -> OCRResult:
        """
        이미지에서 레이아웃과 텍스트 추출

        Args:
            image_path: 이미지 파일 경로
            output_dir: 결과 저장 디렉토리
            progress_callback: 진행률 콜백

        Returns:
            OCRResult: 추출된 레이아웃 정보 (텍스트 + bbox)
        """
        if not PIL_AVAILABLE:
            return OCRResult(
                success=False,
                components=[],
                page_size={},
                confidence=0.0,
                processing_time=0,
                error="Pillow가 설치되지 않았습니다."
            )

        start_time = time.time()
        thread_id = threading.current_thread().ident

        print(f"\n📄 스레드 {thread_id}: DeepSeek-OCR 분석 시작")

        try:
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

            if output_dir is None:
                output_dir = image_path.parent / f"ocr_output_{int(time.time())}"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback(0, 100, "DeepSeek-OCR 모델 로드 중...")

            # 모델 로드
            model, tokenizer = self._get_model_and_tokenizer()

            if progress_callback:
                progress_callback(20, 100, "이미지 분석 중...")

            # 이미지 크기 확인
            img = Image.open(image_path)
            page_size = {
                "width": img.width,
                "height": img.height,
                "dpi": img.info.get('dpi', (200, 200))[0]
            }

            print(f"  📐 이미지 크기: {img.width} x {img.height}px")

            # DeepSeek-OCR 실행 (grounding 모드)
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

            print(f"  🔍 스레드 {thread_id}: OCR 분석 중...")

            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=False
            )

            if progress_callback:
                progress_callback(80, 100, "레이아웃 파싱 중...")

            # 결과 파싱
            components = self._parse_ocr_result(result, page_size)

            # JSON 저장
            json_path = output_dir / f"{image_path.stem}_layout.json"
            layout_data = {
                "page_size": page_size,
                "components": components,
                "metadata": {
                    "source_image": str(image_path),
                    "timestamp": time.time(),
                    "model": self.model_name
                }
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(layout_data, f, ensure_ascii=False, indent=2)

            print(f"  💾 레이아웃 JSON 저장: {json_path}")

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(100, 100, "분석 완료")

            print(f"✅ 스레드 {thread_id}: OCR 분석 완료 ({processing_time:.2f}초)")
            print(f"   컴포넌트 수: {len(components)}개")

            return OCRResult(
                success=True,
                components=components,
                page_size=page_size,
                confidence=0.95,
                processing_time=processing_time,
                metadata={
                    "json_path": str(json_path),
                    "thread_id": thread_id,
                    "component_count": len(components)
                }
            )

        except Exception as e:
            self.logger.error(f"스레드 {thread_id}: OCR 분석 실패: {e}")
            traceback.print_exc()

            return OCRResult(
                success=False,
                components=[],
                page_size={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=str(e)
            )

    def _parse_ocr_result(self, result: Any, page_size: Dict) -> List[Dict[str, Any]]:
        """
        DeepSeek-OCR 결과를 파싱하여 컴포넌트 리스트 생성

        Args:
            result: DeepSeek-OCR 결과
            page_size: 페이지 크기 정보

        Returns:
            List[Dict]: 컴포넌트 리스트 (type, bbox, text, font_size)
        """
        components = []

        try:
            # DeepSeek-OCR 결과 구조 파싱
            # 실제 결과 구조에 맞게 수정 필요
            if isinstance(result, dict):
                # Markdown 텍스트 추출
                markdown_text = result.get('markdown', '')

                # Grounding 정보 (bbox) 추출
                grounding_info = result.get('grounding', [])

                # 각 텍스트 블록을 컴포넌트로 변환
                for idx, item in enumerate(grounding_info):
                    bbox = item.get('bbox', [0, 0, 100, 100])
                    text = item.get('text', '')
                    comp_type = item.get('type', 'text')

                    # bbox 높이로 폰트 크기 추정
                    font_size = self._estimate_font_size(bbox, page_size.get('dpi', 200))

                    component = {
                        "type": comp_type,
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "text": text,
                        "confidence": item.get('confidence', 0.9),
                        "font_size_estimate": font_size,
                        "index": idx
                    }

                    components.append(component)

            # 컴포넌트가 없으면 전체 텍스트를 하나의 컴포넌트로
            if not components and isinstance(result, str):
                components.append({
                    "type": "text",
                    "bbox": [0, 0, page_size['width'], page_size['height']],
                    "text": result,
                    "confidence": 0.9,
                    "font_size_estimate": 12,
                    "index": 0
                })

        except Exception as e:
            self.logger.warning(f"OCR 결과 파싱 실패: {e}")
            # 기본 컴포넌트 생성
            components.append({
                "type": "text",
                "bbox": [0, 0, page_size.get('width', 1000), page_size.get('height', 1000)],
                "text": str(result),
                "confidence": 0.5,
                "font_size_estimate": 12,
                "index": 0
            })

        return components

    def _estimate_font_size(self, bbox: List[int], dpi: int = 200) -> int:
        """
        bbox 높이로 폰트 크기 추정

        Args:
            bbox: [x1, y1, x2, y2] 좌표
            dpi: 이미지 DPI

        Returns:
            int: 추정된 폰트 크기 (포인트)
        """
        if len(bbox) < 4:
            return 12

        x1, y1, x2, y2 = bbox
        height_px = abs(y2 - y1)

        # 픽셀 → 포인트 변환
        height_pt = height_px * 72 / dpi

        # 폰트 크기는 보통 bbox 높이의 70-80%
        font_size_pt = height_pt * 0.75

        # 최소 8pt, 최대 72pt
        font_size_pt = max(8, min(72, font_size_pt))

        return round(font_size_pt)

    def cleanup_resources(self):
        """리소스 정리"""
        thread_id = threading.current_thread().ident

        if hasattr(self._local, 'model') and self._local.model is not None:
            try:
                print(f"🧹 스레드 {thread_id}: DeepSeek-OCR 모델 정리 중...")
                del self._local.model
                del self._local.tokenizer
                self._local.model = None
                self._local.tokenizer = None

                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"✅ 스레드 {thread_id}: 모델 정리 완료")
            except Exception as e:
                self.logger.warning(f"스레드 {thread_id}: 모델 정리 실패: {e}")


def test_deepseek_ocr():
    """테스트 함수"""
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("이미지 파일 경로를 입력하세요: ")

    if not Path(image_path).exists():
        print(f"❌ 파일이 존재하지 않습니다: {image_path}")
        return

    # 서비스 생성
    service = DeepSeekOCRService(config={'use_gpu': True})

    # OCR 실행
    result = service.extract_layout(image_path)

    if result.success:
        print(f"\n✅ OCR 분석 성공!")
        print(f"   컴포넌트 수: {len(result.components)}")
        print(f"   처리 시간: {result.processing_time:.2f}초")
        print(f"\n컴포넌트 목록:")
        for i, comp in enumerate(result.components[:5]):  # 처음 5개만 출력
            print(f"   [{i}] {comp['type']}: {comp['text'][:50]}...")
            print(f"       bbox: {comp['bbox']}, font: {comp['font_size_estimate']}pt")
    else:
        print(f"❌ OCR 분석 실패: {result.error}")

    # 리소스 정리
    service.cleanup_resources()


if __name__ == "__main__":
    test_deepseek_ocr()
