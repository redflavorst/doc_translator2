# services/searchable_pdf_generator.py
"""
검색 가능한 PDF 생성기
이미지 배경 + 투명 텍스트 레이어 = Searchable PDF (샌드위치 PDF)
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF")


class SearchablePDFGenerator:
    """검색 가능한 PDF 생성기"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_searchable_pdf(self, image_path: str, layout_json: Dict,
                              output_path: str, debug_mode: bool = False):
        """
        검색 가능한 PDF 생성 (샌드위치 PDF)

        Args:
            image_path: 배경 이미지 경로
            layout_json: 레이아웃 JSON (components 포함)
            output_path: 출력 PDF 경로
            debug_mode: 디버그 모드 (텍스트 레이어 보이기)
        """
        if not PYMUPDF_AVAILABLE:
            self.logger.error("PyMuPDF not available")
            return False

        try:
            print(f"\n📄 검색 가능한 PDF 생성 시작")
            print(f"   이미지: {image_path}")
            print(f"   출력: {output_path}")

            # PDF 문서 생성
            pdf = fitz.open()

            # 페이지 크기
            page_size = layout_json.get('page_size', {})
            width = page_size.get('width', 1654)
            height = page_size.get('height', 2339)

            # 새 페이지 생성
            page = pdf.new_page(width=width, height=height)

            # 1) 배경 이미지 삽입
            img_rect = fitz.Rect(0, 0, width, height)
            page.insert_image(img_rect, filename=image_path)

            print(f"   ✅ 배경 이미지 삽입 완료")

            # 2) 투명 텍스트 레이어 추가
            components = layout_json.get('components', [])
            text_layer_count = 0

            for comp in components:
                comp_type = comp.get('type', 'text')
                text = comp.get('text', '')
                bbox = comp.get('bbox', [])

                # 텍스트가 있는 컴포넌트만 처리
                if not text or len(bbox) < 4:
                    continue

                # 테이블 셀 처리
                if comp_type == 'table':
                    cells = comp.get('cells', [])
                    for cell in cells:
                        cell_text = cell.get('text', '')
                        cell_bbox = cell.get('bbox', [])
                        if cell_text and len(cell_bbox) >= 4:
                            self._insert_text_layer(
                                page, cell_text, cell_bbox,
                                comp.get('font_size_estimate', 12),
                                debug_mode
                            )
                            text_layer_count += 1
                else:
                    # 일반 텍스트
                    self._insert_text_layer(
                        page, text, bbox,
                        comp.get('font_size_estimate', 12),
                        debug_mode
                    )
                    text_layer_count += 1

            print(f"   ✅ 텍스트 레이어 {text_layer_count}개 추가")

            # PDF 저장
            pdf.save(output_path, garbage=4, deflate=True)
            pdf.close()

            print(f"✅ 검색 가능한 PDF 생성 완료: {output_path}")
            print(f"   텍스트 레이어: {text_layer_count}개")
            print(f"   디버그 모드: {'ON' if debug_mode else 'OFF'}")

            return True

        except Exception as e:
            self.logger.error(f"검색 가능한 PDF 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _insert_text_layer(self, page, text: str, bbox: List[int],
                           font_size: int, debug_mode: bool):
        """
        투명 텍스트 레이어 삽입

        Args:
            page: PyMuPDF 페이지 객체
            text: 텍스트 내용
            bbox: [x1, y1, x2, y2] 좌표
            font_size: 폰트 크기
            debug_mode: 디버그 모드 (텍스트 보이기)
        """
        try:
            x1, y1, x2, y2 = bbox

            # 폰트 크기 조정 (최소 6pt, 최대 72pt)
            font_size = max(6, min(72, font_size))

            # 텍스트 박스 생성
            rect = fitz.Rect(x1, y1, x2, y2)

            if debug_mode:
                # 디버그 모드: 텍스트 보이기 (반투명)
                page.insert_textbox(
                    rect,
                    text,
                    fontsize=font_size,
                    fontname="helv",  # Helvetica
                    fontfile=None,
                    align=fitz.TEXT_ALIGN_LEFT,
                    color=(1, 0, 0),  # 빨간색
                    fill=(1, 1, 1),   # 흰색 배경
                    opacity=0.5,      # 반투명
                    overlay=True
                )
            else:
                # 정상 모드: 투명 텍스트 (보이지 않음)
                page.insert_textbox(
                    rect,
                    text,
                    fontsize=font_size,
                    fontname="helv",
                    fontfile=None,
                    align=fitz.TEXT_ALIGN_LEFT,
                    color=(0, 0, 0),      # 검은색
                    render_mode=3,        # 투명 모드 (neither fill nor stroke)
                    overlay=True
                )

        except Exception as e:
            self.logger.warning(f"텍스트 레이어 삽입 실패: {e}")

    def create_multi_page_searchable_pdf(self, page_data_list: List[Dict],
                                         output_path: str, debug_mode: bool = False):
        """
        다중 페이지 검색 가능한 PDF 생성

        Args:
            page_data_list: 페이지별 데이터 리스트
                [
                    {
                        'image_path': 'page_1.png',
                        'layout_json': {...}
                    },
                    ...
                ]
            output_path: 출력 PDF 경로
            debug_mode: 디버그 모드
        """
        if not PYMUPDF_AVAILABLE:
            self.logger.error("PyMuPDF not available")
            return False

        try:
            print(f"\n📚 다중 페이지 검색 가능한 PDF 생성 시작")
            print(f"   페이지 수: {len(page_data_list)}")
            print(f"   출력: {output_path}")

            # PDF 문서 생성
            pdf = fitz.open()

            for page_idx, page_data in enumerate(page_data_list):
                print(f"\n   📄 페이지 {page_idx + 1}/{len(page_data_list)} 처리 중...")

                image_path = page_data['image_path']
                layout_json = page_data['layout_json']

                # 페이지 크기
                page_size = layout_json.get('page_size', {})
                width = page_size.get('width', 1654)
                height = page_size.get('height', 2339)

                # 새 페이지 생성
                page = pdf.new_page(width=width, height=height)

                # 배경 이미지 삽입
                img_rect = fitz.Rect(0, 0, width, height)
                page.insert_image(img_rect, filename=image_path)

                # 텍스트 레이어 추가
                components = layout_json.get('components', [])
                text_count = 0

                for comp in components:
                    text = comp.get('text', '')
                    bbox = comp.get('bbox', [])
                    comp_type = comp.get('type', 'text')

                    if not text or len(bbox) < 4:
                        continue

                    # 테이블 처리
                    if comp_type == 'table':
                        cells = comp.get('cells', [])
                        for cell in cells:
                            cell_text = cell.get('text', '')
                            cell_bbox = cell.get('bbox', [])
                            if cell_text and len(cell_bbox) >= 4:
                                self._insert_text_layer(
                                    page, cell_text, cell_bbox,
                                    cell.get('font_size_estimate', 12),
                                    debug_mode
                                )
                                text_count += 1
                    else:
                        self._insert_text_layer(
                            page, text, bbox,
                            comp.get('font_size_estimate', 12),
                            debug_mode
                        )
                        text_count += 1

                print(f"      ✅ 텍스트 레이어 {text_count}개 추가")

            # PDF 저장
            pdf.save(output_path, garbage=4, deflate=True)
            pdf.close()

            print(f"\n✅ 다중 페이지 검색 가능한 PDF 생성 완료!")
            print(f"   출력 파일: {output_path}")
            print(f"   총 페이지: {len(page_data_list)}개")

            return True

        except Exception as e:
            self.logger.error(f"다중 페이지 PDF 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_searchable_pdf():
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

    # 검색 가능한 PDF 생성
    generator = SearchablePDFGenerator()
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_searchable.pdf"

    # 일반 모드
    success = generator.create_searchable_pdf(
        image_path=image_path,
        layout_json=layout_json,
        output_path=str(output_path),
        debug_mode=False
    )

    # 디버그 모드 PDF도 생성
    if success:
        debug_output = Path(image_path).parent / f"{Path(image_path).stem}_searchable_debug.pdf"
        generator.create_searchable_pdf(
            image_path=image_path,
            layout_json=layout_json,
            output_path=str(debug_output),
            debug_mode=True
        )
        print(f"\n✅ 디버그 PDF도 생성됨: {debug_output}")


if __name__ == "__main__":
    test_searchable_pdf()
