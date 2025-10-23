# DeepSeek-OCR 기반 번역 시스템

PaddleOCR을 DeepSeek-OCR로 대체한 새로운 번역 파이프라인입니다.

## 주요 특징

✅ **검색 가능한 PDF** - 원본과 번역본 모두 Ctrl+F 검색 가능
✅ **레이아웃 보존** - 원본 위치 그대로 번역 텍스트 배치 (DEEPL 스타일)
✅ **디버그 모드** - bbox 테두리로 컴포넌트 시각적 확인
✅ **GPU 가속** - DeepSeek-OCR은 단일 GPU로 하루 20만 페이지 처리

## 출력 파일

1. **원본 검색 가능 PDF** - 원문 텍스트 검색/복사 가능
2. **번역 검색 가능 PDF** - 한글 번역 텍스트 검색/복사 가능
3. **디버그 PDF** (옵션) - bbox 테두리 표시로 컴포넌트 확인

## 설치

```bash
# DeepSeek-OCR 의존성 설치
pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3
pip install einops addict easydict

# 기존 의존성
pip install PyMuPDF Pillow

# 옵션: flash-attention (GPU 가속)
pip install flash-attn==2.7.3 --no-build-isolation
```

## 사용법

### 1. 기본 사용 (스탠드얼론)

```bash
python deepseek_translation_workflow.py input.pdf
```

### 2. 프로그래밍 방식

```python
from deepseek_translation_workflow import DeepSeekTranslationWorkflow

# 워크플로우 초기화
workflow = DeepSeekTranslationWorkflow(use_gpu=True)

# PDF 처리
result = workflow.process_pdf(
    pdf_path="input.pdf",
    output_dir="output",
    debug_mode=True  # 디버그 PDF 생성
)

# 결과
print(f"원본 PDF: {result['original_pdf']}")
print(f"번역 PDF: {result['translated_pdf']}")

# 리소스 정리
workflow.cleanup()
```

## 파이프라인 구조

```
PDF 입력
  ↓
[1] 페이지별 이미지 추출 (PyMuPDF, 200 DPI)
  ↓
[2] DeepSeek-OCR 레이아웃 분석
     → JSON (텍스트 + bbox 좌표)
  ↓
[3] 디버그 시각화 (옵션)
     → bbox 테두리 이미지
  ↓
[4] Ollama LLM 번역
     → 번역된 JSON
  ↓
[5] 이미지 오버레이 렌더링
     → 원본 영역 마스킹 + 한글 렌더링
  ↓
[6] 검색 가능한 PDF 생성
     → 배경 이미지 + 투명 텍스트 레이어
  ↓
출력: 원본 PDF + 번역 PDF
```

## 주요 서비스

### 1. DeepSeekOCRService
```python
from services.deepseek_ocr_service import DeepSeekOCRService

service = DeepSeekOCRService(config={'use_gpu': True})
result = service.extract_layout(image_path="page.png")

# 결과: 컴포넌트 리스트 (type, bbox, text, font_size)
for comp in result.components:
    print(f"{comp['type']}: {comp['text'][:50]}")
    print(f"  bbox: {comp['bbox']}, font: {comp['font_size_estimate']}pt")
```

### 2. DebugVisualizer
```python
from services.debug_visualizer import DebugVisualizer

visualizer = DebugVisualizer()
visualizer.draw_debug_borders(
    image_path="page.png",
    layout_json=layout_json,
    output_path="debug.png",
    show_labels=True
)
```

### 3. TranslationService (JSON 기반)
```python
from services.translation_service import TranslationService

service = TranslationService()
translated_json = service.translate_layout_json(
    layout_json=layout_json,
    output_path="translated.json"
)
```

### 4. ImageOverlayRenderer
```python
from services.image_overlay_renderer import ImageOverlayRenderer

renderer = ImageOverlayRenderer()
renderer.render_translation(
    original_image_path="page.png",
    translated_json=translated_json,
    output_path="rendered.png"
)
```

### 5. SearchablePDFGenerator
```python
from services.searchable_pdf_generator import SearchablePDFGenerator

generator = SearchablePDFGenerator()
generator.create_searchable_pdf(
    image_path="rendered.png",
    layout_json=translated_json,
    output_path="output.pdf",
    debug_mode=False  # 투명 텍스트 레이어
)
```

## 검색 가능한 PDF 구조

```
Page 1:
  ┌─────────────────────────┐
  │ [배경 이미지 레이어]        │ ← 사용자가 보는 것
  │ ┌─────────────────┐     │   (한글 폰트로 렌더링)
  │ │  소개            │     │
  │ │  이 문서는...     │     │
  │ └─────────────────┘     │
  │                         │
  │ [투명 텍스트 레이어]       │ ← 보이지 않지만 검색/복사 가능
  │ ┌─────────────────┐     │   (정확한 bbox 위치)
  │ │ 소개 (투명)       │     │
  │ │ 이 문서는... (투명)│     │
  │ └─────────────────┘     │
  └─────────────────────────┘
```

## 출력 디렉토리 구조

```
deepseek_output_XXXXXXXXXX/
├── images/                      # 페이지 이미지
│   ├── page_0001.png
│   └── page_0002.png
├── layouts/                     # 레이아웃 JSON
│   ├── page_0001_layout.json
│   └── page_0002_layout.json
├── translated/                  # 번역 JSON
│   ├── page_0001_translated.json
│   └── page_0002_translated.json
├── rendered/                    # 렌더링 이미지
│   ├── page_0001_rendered.png
│   └── page_0002_rendered.png
├── debug/                       # 디버그 이미지 (bbox 표시)
│   ├── page_0001_debug.png
│   └── page_0002_debug.png
├── input_original_searchable.pdf    # 원본 검색 가능 PDF
├── input_original_debug.pdf         # 원본 디버그 PDF
└── input_translated_searchable.pdf  # 번역 검색 가능 PDF
```

## 디버그 모드

디버그 모드를 활성화하면 각 컴포넌트에 색상별 테두리가 표시됩니다:

- 🟢 **초록**: 일반 텍스트
- 🟡 **노랑**: 헤더
- 🟠 **주황**: 푸터
- 🔴 **빨강**: 제목
- 🔵 **파랑**: 테이블
- 🟣 **보라**: 리스트
- 🩷 **자홍**: 이미지

## 성능

- **DeepSeek-OCR**: 단일 A100 GPU로 하루 20만 페이지
- **텍스트 압축**: 7~20배 토큰 절감
- **정확도**: 97% 디코딩 정확도

## 기존 PaddleOCR 대비 장점

| 항목 | PaddleOCR | DeepSeek-OCR |
|------|-----------|--------------|
| GPU 효율 | 낮음 | 매우 높음 (20만 페이지/일) |
| 레이아웃 보존 | 보통 | 우수 |
| 마크다운 변환 | 수동 | 자동 |
| 테이블 인식 | 별도 모델 | 통합 |
| 모델 크기 | 여러 모델 | 3B 단일 모델 |

## 문제 해결

### 1. CUDA Out of Memory
```python
# GPU 메모리 부족 시 CPU 모드 사용
workflow = DeepSeekTranslationWorkflow(use_gpu=False)
```

### 2. 한글 폰트 없음
```bash
# Windows: malgun.ttf 자동 사용
# Linux: NanumGothic 설치
sudo apt-get install fonts-nanum

# macOS: AppleSDGothicNeo 자동 사용
```

### 3. DeepSeek-OCR 모델 다운로드 느림
```bash
# Hugging Face 미러 사용
export HF_ENDPOINT=https://hf-mirror.com
```

## 참고

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [PyMuPDF 문서](https://pymupdf.readthedocs.io/)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
