# services/image_overlay_renderer.py
"""
이미지 오버레이 렌더러
원본 이미지에 번역 텍스트를 렌더링 (DEEPL 스타일)
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import textwrap

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")


class ImageOverlayRenderer:
    """이미지 오버레이 렌더러"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 한글 폰트 경로 시도 목록
        self.korean_font_paths = [
            # Windows
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/gulim.ttc",
            # Linux
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            # macOS
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/NanumGothic.ttf",
        ]

    def render_translation(self, original_image_path: str,
                          translated_json: Dict,
                          output_path: str) -> bool:
        """
        번역된 텍스트를 이미지에 렌더링

        Args:
            original_image_path: 원본 이미지 경로
            translated_json: 번역된 레이아웃 JSON
            output_path: 출력 이미지 경로

        Returns:
            bool: 성공 여부
        """
        if not PIL_AVAILABLE:
            self.logger.error("PIL not available")
            return False

        try:
            print(f"\n🎨 번역 이미지 렌더링 시작")
            print(f"   원본 이미지: {original_image_path}")
            print(f"   출력: {output_path}")

            # 이미지 로드
            img = Image.open(original_image_path).convert('RGB')
            draw = ImageDraw.Draw(img)

            # 컴포넌트 처리
            components = translated_json.get('components', [])
            rendered_count = 0

            for comp in components:
                comp_type = comp.get('type', 'text')
                translated_text = comp.get('translated_text', '')
                bbox = comp.get('bbox', [])
                font_size = comp.get('font_size_estimate', 12)

                # 번역된 텍스트가 없으면 스킵
                if not translated_text or len(bbox) < 4:
                    continue

                # 테이블 처리
                if comp_type == 'table':
                    cells = comp.get('cells', [])
                    for cell in cells:
                        cell_translated = cell.get('translated_text', '')
                        cell_bbox = cell.get('bbox', [])
                        cell_font = cell.get('font_size_estimate', 10)

                        if cell_translated and len(cell_bbox) >= 4:
                            self._render_text_box(
                                draw, img, cell_translated, cell_bbox, cell_font
                            )
                            rendered_count += 1
                else:
                    # 일반 텍스트
                    self._render_text_box(
                        draw, img, translated_text, bbox, font_size
                    )
                    rendered_count += 1

            # 이미지 저장
            img.save(output_path, quality=95)

            print(f"✅ 번역 이미지 렌더링 완료: {output_path}")
            print(f"   렌더링된 텍스트 박스: {rendered_count}개")

            return True

        except Exception as e:
            self.logger.error(f"렌더링 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _render_text_box(self, draw, img, text: str, bbox: List[int],
                         font_size: int):
        """
        텍스트 박스 렌더링

        Args:
            draw: ImageDraw 객체
            img: Image 객체
            text: 텍스트 내용
            bbox: [x1, y1, x2, y2] 좌표
            font_size: 폰트 크기
        """
        try:
            x1, y1, x2, y2 = bbox

            # 좌표 유효성 검사
            x1 = max(0, min(x1, img.width))
            x2 = max(0, min(x2, img.width))
            y1 = max(0, min(y1, img.height))
            y2 = max(0, min(y2, img.height))

            # 박스가 너무 작으면 스킵
            if abs(x2 - x1) < 10 or abs(y2 - y1) < 5:
                return

            # 1) 원본 텍스트 영역 마스킹 (흰색 사각형)
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

            # 2) 번역 텍스트 렌더링
            # 폰트 로드
            font = self._get_korean_font(font_size)

            # 텍스트 줄바꿈 처리
            box_width = x2 - x1
            box_height = y2 - y1

            # 한 줄당 글자 수 추정
            avg_char_width = font_size * 0.6  # 한글 평균 너비
            chars_per_line = max(1, int(box_width / avg_char_width))

            # 텍스트 줄바꿈
            wrapped_lines = textwrap.wrap(text, width=chars_per_line)

            # 줄 수 제한 (박스 높이에 맞춤)
            line_height = font_size * 1.2
            max_lines = max(1, int(box_height / line_height))
            wrapped_lines = wrapped_lines[:max_lines]

            # 텍스트 그리기
            y_offset = y1
            for line in wrapped_lines:
                if y_offset + line_height > y2:
                    break

                # 텍스트 그리기
                draw.text(
                    (x1 + 2, y_offset),
                    line,
                    fill=(0, 0, 0),
                    font=font
                )

                y_offset += line_height

        except Exception as e:
            self.logger.warning(f"텍스트 박스 렌더링 실패: {e}")

    def _get_korean_font(self, size: int):
        """한글 폰트 로드"""
        size = max(8, min(72, size))

        # 한글 폰트 시도
        for font_path in self.korean_font_paths:
            try:
                if Path(font_path).exists():
                    return ImageFont.truetype(font_path, size)
            except:
                continue

        # 기본 폰트로 fallback
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()


def test_renderer():
    """테스트 함수"""
    import sys
    import json

    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        json_path = sys.argv[2]
    else:
        image_path = input("원본 이미지 경로: ")
        json_path = input("번역된 JSON 경로: ")

    if not Path(image_path).exists():
        print(f"❌ 이미지 파일이 존재하지 않습니다: {image_path}")
        return

    if not Path(json_path).exists():
        print(f"❌ JSON 파일이 존재하지 않습니다: {json_path}")
        return

    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        translated_json = json.load(f)

    # 렌더링
    renderer = ImageOverlayRenderer()
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_translated.png"

    success = renderer.render_translation(
        original_image_path=image_path,
        translated_json=translated_json,
        output_path=str(output_path)
    )

    if success:
        print(f"\n✅ 번역 이미지 생성 완료: {output_path}")


if __name__ == "__main__":
    test_renderer()
