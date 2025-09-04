# PDF Document Translator

PDF 문서를 한국어로 번역하는 서비스입니다. PaddleOCR을 사용하여 PDF 레이아웃을 분석하고, Ollama를 통해 한국어로 번역합니다.

## 🚀 주요 기능

- **PDF 레이아웃 분석**: PaddleOCR PPStructureV3를 사용한 정확한 문서 구조 분석
- **페이지별 실시간 진행률 추적**: 각 페이지의 레이아웃 분석 및 번역 진행 상황 실시간 확인
- **한국어 번역**: Ollama LLM을 활용한 고품질 한국어 번역
- **웹 인터페이스**: 직관적인 웹 UI를 통한 파일 업로드 및 진행 상황 모니터링
- **워크플로우 관리**: 비동기 처리 및 상태 추적

## 📋 필요 사항

- Python 3.8+
- Ollama (로컬에서 실행 중)
- PaddleOCR 지원 환경

## 🛠️ 설치

1. 저장소 클론
```bash
git clone https://github.com/yourusername/doc_translator.git
cd doc_translator
```

2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 설정 값 수정
```

5. Ollama 모델 설치
```bash
ollama pull gemma3n:e4b
```

## 🚀 실행

### 웹 서버 실행
```bash
python run_server.py
```

브라우저에서 `http://localhost:8000` 접속

### API 직접 사용
```python
from services.layout_analysis_service_paged import LayoutAnalysisServicePaged
from services.translation_service import TranslationService

# 레이아웃 분석
layout_service = LayoutAnalysisServicePaged()
result = layout_service.analyze_document("document.pdf", "output_dir")

# 번역
translation_service = TranslationService()
translated = translation_service.translate_document("page_001.md")
```

## 📁 프로젝트 구조

```
doc_translator/
├── api/                    # FastAPI 웹 서버
│   └── main.py            # API 엔드포인트 및 웹 UI
├── core/                  # 핵심 비즈니스 로직
│   ├── config.py          # 설정 관리
│   ├── models.py          # 데이터 모델
│   ├── workflow_manager.py # 워크플로우 관리
│   └── error_handler.py   # 에러 처리
├── services/              # 서비스 레이어
│   ├── layout_analysis_service.py       # 기본 레이아웃 분석
│   ├── layout_analysis_service_paged.py # 페이지별 분석 (실시간 진행률)
│   └── translation_service.py           # 번역 서비스
├── tests/                 # 테스트 코드
├── requirements.txt       # Python 의존성
├── .env.example          # 환경 변수 예시
└── run_server.py         # 서버 실행 스크립트
```

## 🔧 설정

`.env` 파일에서 다음 설정을 조정할 수 있습니다:

- `OLLAMA_BASE_URL`: Ollama 서버 주소 (기본: http://localhost:11434)
- `TRANSLATION_MODEL`: 번역 모델 (기본: gemma3n:e4b)
- `API_PORT`: 웹 서버 포트 (기본: 8000)
- `MAX_CONCURRENT_WORKFLOWS`: 동시 처리 가능한 워크플로우 수

## 📈 진행률 계산 방식

전체 진행률은 다음과 같이 계산됩니다:

```
총 작업 수 = 페이지 수 × 2 (레이아웃 분석 + 번역)
진행률 = (완료된 작업 수 / 총 작업 수) × 100
```

예: 8페이지 문서
- 레이아웃 분석 완료: 8/16 = 50%
- 번역 완료: 16/16 = 100%

## 🤝 기여

기여를 환영합니다! Pull Request를 보내주세요.

## 📝 라이선스

MIT License

## 🙏 감사의 말

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR 및 레이아웃 분석
- [Ollama](https://ollama.ai/) - 로컬 LLM 실행
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크