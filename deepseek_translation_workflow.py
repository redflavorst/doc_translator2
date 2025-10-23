#!/usr/bin/env python3
# deepseek_translation_workflow.py
"""
DeepSeek-OCR 기반 번역 워크플로우
PDF → 검색 가능한 원본 PDF + 검색 가능한 번역 PDF
"""

import sys
import time
import json
from pathlib import Path

# 서비스 임포트
try:
    import fitz  # PyMuPDF
    from services.deepseek_ocr_service import DeepSeekOCRService
    from services.translation_service import TranslationService
    from services.debug_visualizer import DebugVisualizer
    from services.image_overlay_renderer import ImageOverlayRenderer
    from services.searchable_pdf_generator import SearchablePDFGenerator
except ImportError as e:
    print(f"❌ 필수 패키지 임포트 실패: {e}")
    print("다음 명령어로 설치하세요:")
    print("pip install PyMuPDF torch transformers Pillow")
    sys.exit(1)


class DeepSeekTranslationWorkflow:
    """DeepSeek-OCR 기반 번역 워크플로우"""

    def __init__(self, use_gpu: bool = True):
        print("\n" + "="*60)
        print("DeepSeek-OCR 번역 워크플로우 초기화")
        print("="*60)

        # 서비스 초기화
        self.ocr_service = DeepSeekOCRService(config={'use_gpu': use_gpu})
        self.translation_service = TranslationService()
        self.visualizer = DebugVisualizer()
        self.renderer = ImageOverlayRenderer()
        self.pdf_generator = SearchablePDFGenerator()

    def process_pdf(self, pdf_path: str, output_dir: str = None,
                   debug_mode: bool = True):
        """
        PDF 처리 (전체 워크플로우)

        Args:
            pdf_path: 입력 PDF 경로
            output_dir: 출력 디렉토리
            debug_mode: 디버그 모드 (bbox 테두리 표시)

        Returns:
            Dict: 결과 정보
        """
        start_time = time.time()
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            print(f"❌ PDF 파일이 존재하지 않습니다: {pdf_path}")
            return None

        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = pdf_path.parent / f"deepseek_output_{int(time.time())}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📄 PDF 처리 시작: {pdf_path.name}")
        print(f"   출력 디렉토리: {output_dir}")

        # 디렉토리 생성
        images_dir = output_dir / "images"
        layouts_dir = output_dir / "layouts"
        translated_dir = output_dir / "translated"
        rendered_dir = output_dir / "rendered"
        debug_dir = output_dir / "debug"

        for dir_path in [images_dir, layouts_dir, translated_dir, rendered_dir, debug_dir]:
            dir_path.mkdir(exist_ok=True)

        # PDF 페이지별 처리
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        page_data_list = []

        print(f"\n📚 총 {total_pages}개 페이지 처리")

        for page_num in range(total_pages):
            print(f"\n{'='*60}")
            print(f"📄 페이지 {page_num + 1}/{total_pages} 처리 중...")
            print(f"{'='*60}")

            # 1) 페이지 → 이미지 추출
            page = doc[page_num]
            zoom = 2.0  # 200 DPI
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)

            image_path = images_dir / f"page_{page_num + 1:04d}.png"
            pix.save(str(image_path))
            print(f"✅ [1/6] 이미지 추출 완료: {image_path.name}")

            # 2) DeepSeek-OCR로 레이아웃 분석
            print(f"🔍 [2/6] DeepSeek-OCR 분석 중...")
            ocr_result = self.ocr_service.extract_layout(
                image_path=str(image_path),
                output_dir=str(layouts_dir)
            )

            if not ocr_result.success:
                print(f"❌ OCR 분석 실패: {ocr_result.error}")
                continue

            layout_json_path = layouts_dir / f"page_{page_num + 1:04d}_layout.json"
            with open(layout_json_path, 'r', encoding='utf-8') as f:
                layout_json = json.load(f)

            print(f"✅ [2/6] OCR 분석 완료: {len(layout_json['components'])}개 컴포넌트")

            # 3) 디버그 시각화 (옵션)
            if debug_mode:
                print(f"🎨 [3/6] 디버그 시각화 생성 중...")
                debug_image_path = debug_dir / f"page_{page_num + 1:04d}_debug.png"
                self.visualizer.draw_debug_borders(
                    image_path=str(image_path),
                    layout_json=layout_json,
                    output_path=str(debug_image_path),
                    show_labels=True
                )
                print(f"✅ [3/6] 디버그 이미지 생성 완료")

            # 4) 번역
            print(f"📝 [4/6] 번역 중...")
            translated_json_path = translated_dir / f"page_{page_num + 1:04d}_translated.json"
            translated_json = self.translation_service.translate_layout_json(
                layout_json=layout_json,
                output_path=str(translated_json_path)
            )
            print(f"✅ [4/6] 번역 완료")

            # 5) 번역 이미지 렌더링
            print(f"🎨 [5/6] 번역 이미지 렌더링 중...")
            rendered_image_path = rendered_dir / f"page_{page_num + 1:04d}_rendered.png"
            self.renderer.render_translation(
                original_image_path=str(image_path),
                translated_json=translated_json,
                output_path=str(rendered_image_path)
            )
            print(f"✅ [5/6] 렌더링 완료")

            # 페이지 데이터 저장
            page_data_list.append({
                'page_num': page_num + 1,
                'original_image': str(image_path),
                'rendered_image': str(rendered_image_path),
                'layout_json': layout_json,
                'translated_json': translated_json
            })

        doc.close()

        print(f"\n{'='*60}")
        print(f"📚 전체 페이지 처리 완료!")
        print(f"{'='*60}")

        # 6) 검색 가능한 PDF 생성 (원본 + 번역본)
        print(f"\n📄 [6/6] 검색 가능한 PDF 생성 중...")

        # 6-1) 원본 검색 가능 PDF
        original_pdf_path = output_dir / f"{pdf_path.stem}_original_searchable.pdf"
        original_debug_pdf_path = output_dir / f"{pdf_path.stem}_original_debug.pdf"

        self.pdf_generator.create_multi_page_searchable_pdf(
            page_data_list=[{
                'image_path': page_data['original_image'],
                'layout_json': page_data['layout_json']
            } for page_data in page_data_list],
            output_path=str(original_pdf_path),
            debug_mode=False
        )

        # 원본 디버그 PDF (테두리 표시)
        if debug_mode:
            self.pdf_generator.create_multi_page_searchable_pdf(
                page_data_list=[{
                    'image_path': str(debug_dir / f"page_{page_data['page_num']:04d}_debug.png"),
                    'layout_json': page_data['layout_json']
                } for page_data in page_data_list],
                output_path=str(original_debug_pdf_path),
                debug_mode=True
            )

        # 6-2) 번역 검색 가능 PDF
        translated_pdf_path = output_dir / f"{pdf_path.stem}_translated_searchable.pdf"

        self.pdf_generator.create_multi_page_searchable_pdf(
            page_data_list=[{
                'image_path': page_data['rendered_image'],
                'layout_json': page_data['translated_json']
            } for page_data in page_data_list],
            output_path=str(translated_pdf_path),
            debug_mode=False
        )

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"✅ 전체 워크플로우 완료!")
        print(f"{'='*60}")
        print(f"⏱️  소요 시간: {elapsed:.2f}초")
        print(f"📄 총 페이지: {total_pages}개")
        print(f"\n📁 출력 파일:")
        print(f"   1️⃣  원본 검색 가능 PDF: {original_pdf_path}")
        if debug_mode:
            print(f"   🐛 원본 디버그 PDF: {original_debug_pdf_path}")
        print(f"   2️⃣  번역 검색 가능 PDF: {translated_pdf_path}")
        print(f"\n📂 출력 디렉토리: {output_dir}")

        # 결과 반환
        return {
            'success': True,
            'total_pages': total_pages,
            'processing_time': elapsed,
            'output_dir': str(output_dir),
            'original_pdf': str(original_pdf_path),
            'translated_pdf': str(translated_pdf_path),
            'debug_pdf': str(original_debug_pdf_path) if debug_mode else None
        }

    def cleanup(self):
        """리소스 정리"""
        print("\n🧹 리소스 정리 중...")
        self.ocr_service.cleanup_resources()
        print("✅ 리소스 정리 완료")


def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("PDF 파일 경로를 입력하세요: ")

    if not Path(pdf_path).exists():
        print(f"❌ 파일이 존재하지 않습니다: {pdf_path}")
        return

    # 워크플로우 실행
    workflow = DeepSeekTranslationWorkflow(use_gpu=True)

    try:
        result = workflow.process_pdf(
            pdf_path=pdf_path,
            debug_mode=True  # 디버그 모드 ON
        )

        if result and result['success']:
            print(f"\n🎉 번역 성공!")
            print(f"\n📖 사용법:")
            print(f"   - 원본 PDF 열기: {result['original_pdf']}")
            print(f"   - 번역 PDF 열기: {result['translated_pdf']}")
            print(f"   - Ctrl+F로 텍스트 검색 가능!")

    finally:
        workflow.cleanup()


if __name__ == "__main__":
    main()
