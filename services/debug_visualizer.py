# services/debug_visualizer.py
"""
디버그용 시각화 서비스
레이아웃 컴포넌트의 bbox를 시각적으로 표시
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")


class DebugVisualizer:
    """디버그용 bbox 시각화"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 컴포넌트 타입별 색상
        self.colors = {
            'text': (0, 255, 0),           # 초록: 일반 텍스트
            'text_line': (0, 200, 0),      # 연두: 텍스트 라인
            'header': (255, 255, 0),       # 노랑: 헤더
            'footer': (255, 128, 0),       # 주황: 푸터
            'title': (255, 0, 0),          # 빨강: 제목
            'table': (0, 0, 255),          # 파랑: 테이블
            'table_cell': (0, 150, 255),   # 하늘: 테이블 셀
            'image': (255, 0, 255),        # 자홍: 이미지
            'figure': (255, 0, 255),       # 자홍: 그림
            'list': (128, 0, 255),         # 보라: 리스트
            'equation': (0, 255, 255),     # 청록: 수식
            'default': (128, 128, 128)     # 회색: 기본
        }

    def draw_debug_borders(self, image_path: str, layout_json: Dict,
                          output_path: str, show_labels: bool = True):
        """
        디버그용 bbox 테두리 그리기

        Args:
            image_path: 원본 이미지 경로
            layout_json: 레이아웃 JSON (components 포함)
            output_path: 출력 이미지 경로
            show_labels: 라벨 표시 여부
        """
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available, skipping visualization")
            return

        try:
            # 이미지 로드
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img, 'RGBA')

            # 폰트 설정
            font = self._get_font(size=20)
            small_font = self._get_font(size=14)

            components = layout_json.get('components', [])

            print(f"\n🎨 디버그 시각화 시작")
            print(f"   컴포넌트 수: {len(components)}")

            # 타입별 카운트
            type_counts = {}

            # 각 컴포넌트 그리기
            for idx, comp in enumerate(components):
                bbox = comp.get('bbox', [0, 0, 100, 100])
                comp_type = comp.get('type', 'default')
                text = comp.get('text', '')
                confidence = comp.get('confidence', 0.0)
                font_size = comp.get('font_size_estimate', 12)

                # 타입 카운트
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

                # 색상 선택
                color = self.colors.get(comp_type, self.colors['default'])

                # bbox 좌표 정규화
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]

                    # 좌표 유효성 검사
                    x1 = max(0, min(x1, img.width))
                    x2 = max(0, min(x2, img.width))
                    y1 = max(0, min(y1, img.height))
                    y2 = max(0, min(y2, img.height))

                    # 박스가 너무 작으면 스킵
                    if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                        continue

                    # 테두리 그리기 (두께 3px)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # 라벨 표시
                    if show_labels:
                        # 라벨 텍스트 생성
                        label = f"#{idx} {comp_type}"
                        if font_size:
                            label += f" ({font_size}pt)"

                        # 텍스트 미리보기 (20자까지)
                        if text:
                            preview = text.replace('\n', ' ')[:20]
                            if len(text) > 20:
                                preview += "..."
                            label += f": {preview}"

                        # 라벨 위치 계산 (박스 위쪽)
                        label_y = max(5, y1 - 25)

                        # 라벨 배경
                        text_bbox = draw.textbbox((x1, label_y), label, font=small_font)
                        padding = 2
                        bg_bbox = [
                            text_bbox[0] - padding,
                            text_bbox[1] - padding,
                            text_bbox[2] + padding,
                            text_bbox[3] + padding
                        ]
                        draw.rectangle(bg_bbox, fill=(255, 255, 255, 230), outline=color)

                        # 라벨 텍스트
                        draw.text((x1, label_y), label, fill=color, font=small_font)

            # 범례 추가
            self._draw_legend(draw, img, type_counts, font)

            # 통계 정보 추가
            self._draw_stats(draw, img, components, font)

            # 이미지 저장
            img.save(output_path, quality=95)

            print(f"✅ 디버그 이미지 저장: {output_path}")
            print(f"   타입별 컴포넌트:")
            for comp_type, count in sorted(type_counts.items()):
                print(f"     - {comp_type}: {count}개")

        except Exception as e:
            self.logger.error(f"시각화 실패: {e}")
            import traceback
            traceback.print_exc()

    def _get_font(self, size: int = 20):
        """폰트 로드 (한글 지원)"""
        try:
            # Windows
            return ImageFont.truetype("malgun.ttf", size)
        except:
            try:
                # Linux
                return ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", size)
            except:
                try:
                    # macOS
                    return ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", size)
                except:
                    try:
                        # 기본 폰트
                        return ImageFont.truetype("arial.ttf", size)
                    except:
                        # 최후의 수단
                        return ImageFont.load_default()

    def _draw_legend(self, draw, img, type_counts: Dict, font):
        """범례 그리기"""
        if not type_counts:
            return

        # 범례 위치 (우측 상단)
        legend_x = img.width - 300
        legend_y = 10
        legend_width = 280
        legend_height = len(type_counts) * 30 + 40

        # 범례 배경
        draw.rectangle(
            [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
            fill=(255, 255, 255, 230),
            outline=(0, 0, 0)
        )

        # 제목
        draw.text((legend_x + 10, legend_y + 10), "컴포넌트 타입", fill='black', font=font)

        # 각 타입별 범례
        y_offset = legend_y + 35
        for comp_type, count in sorted(type_counts.items()):
            color = self.colors.get(comp_type, self.colors['default'])

            # 색상 박스
            draw.rectangle(
                [legend_x + 10, y_offset, legend_x + 30, y_offset + 20],
                fill=color,
                outline='black'
            )

            # 라벨
            label = f"{comp_type}: {count}개"
            draw.text((legend_x + 40, y_offset), label, fill='black', font=font)

            y_offset += 25

    def _draw_stats(self, draw, img, components: List, font):
        """통계 정보 그리기 (하단)"""
        stats_text = f"총 {len(components)}개 컴포넌트 검출"

        # 배경
        text_bbox = draw.textbbox((15, img.height - 35), stats_text, font=font)
        padding = 5
        bg_bbox = [
            text_bbox[0] - padding,
            text_bbox[1] - padding,
            text_bbox[2] + padding,
            text_bbox[3] + padding
        ]
        draw.rectangle(bg_bbox, fill=(255, 255, 255, 230), outline='black')

        # 텍스트
        draw.text((15, img.height - 35), stats_text, fill='black', font=font)


def test_visualizer():
    """테스트 함수"""
    import sys
    import json

    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        json_path = sys.argv[2]
    else:
        image_path = input("이미지 파일 경로: ")
        json_path = input("레이아웃 JSON 경로: ")

    if not Path(image_path).exists():
        print(f"❌ 이미지 파일이 존재하지 않습니다: {image_path}")
        return

    if not Path(json_path).exists():
        print(f"❌ JSON 파일이 존재하지 않습니다: {json_path}")
        return

    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        layout_json = json.load(f)

    # 시각화
    visualizer = DebugVisualizer()
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_debug.png"

    visualizer.draw_debug_borders(
        image_path=image_path,
        layout_json=layout_json,
        output_path=str(output_path),
        show_labels=True
    )

    print(f"\n✅ 디버그 이미지 생성 완료: {output_path}")


if __name__ == "__main__":
    test_visualizer()
