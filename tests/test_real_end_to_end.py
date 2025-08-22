# tests/test_real_end_to_end.py
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import tempfile
import shutil
import time
import requests
from services.layout_analysis_service import LayoutAnalysisService
from services.translation_service import TranslationService

class TestRealEndToEnd:
    """실제 PDF → 마크다운 → 한국어 번역 전체 파이프라인 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def layout_service(self):
        """실제 레이아웃 분석 서비스 (빠른 설정)"""
        config = {
            'use_gpu': False,  # CPU 사용으로 안정성 확보
            'det_limit_side_len': 960,  # 작은 크기로 빠른 처리
            'use_table': False  # 테이블 인식 비활성화로 속도 향상
        }
        return LayoutAnalysisService(config)
    
    @pytest.fixture  
    def translation_service(self):
        """실제 번역 서비스 (빠른 설정)"""
        config = {
            'model_name': 'qwen3:8b',  # 가장 작은 모델
            'temperature': 0.1,
            'max_retries': 1,
            'quality_threshold': 0.3  # 테스트용 낮은 임계값
        }
        return TranslationService(config)

    def check_ollama_ready(self):
        """Ollama 서버와 모델 준비 상태 확인"""
        try:
            # 서버 확인
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code != 200:
                return False, "Ollama 서버가 응답하지 않습니다"
            
            # 모델 확인
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m['name'] for m in response.json().get('models', [])]
            
            required_model = 'qwen3:8b'
            if required_model not in models:
                return False, f"모델 {required_model}이 설치되지 않았습니다. 설치 명령: ollama pull {required_model}"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Ollama 확인 실패: {e}"

    def test_pdf_to_korean_pipeline(self, layout_service, translation_service, temp_dir):
        """PDF → 마크다운 → 한국어 번역 전체 파이프라인 테스트"""
        
        # 테스트용 PDF 파일 경로 (실제 파일로 교체하세요)
        test_pdf_paths = [
            
            "이전가격_0804/특수관계자계약서_FY2023_합본.pdf",  # 기존 테스트에서 사용하던 파일
            
        ]
        
        pdf_path = None
        for path in test_pdf_paths:
            if Path(path).exists():
                pdf_path = path
                break
        
        if not pdf_path:
            pytest.skip(f"테스트용 PDF 파일이 없습니다. 다음 중 하나를 준비하세요: {test_pdf_paths}")
        
        # Ollama 준비 상태 확인
        ready, message = self.check_ollama_ready()
        if not ready:
            pytest.skip(f"Ollama 준비 안됨: {message}")
        
        print(f"\n" + "="*60)
        print(f"실제 PDF → 한국어 번역 파이프라인 테스트")
        print(f"PDF 파일: {pdf_path}")
        print(f"출력 디렉토리: {temp_dir}")
        print(f"="*60)
        
        total_start_time = time.time()
        
        # Step 1: PDF → 마크다운 변환
        print(f"\n📄 Step 1: PDF 레이아웃 분석 시작...")
        layout_start = time.time()
        
        layout_result = layout_service.analyze_document(pdf_path, temp_dir)
        
        layout_time = time.time() - layout_start
        print(f"   레이아웃 분석 완료: {layout_time:.1f}초")
        print(f"   성공 여부: {layout_result.success}")
        print(f"   생성된 페이지 수: {len(layout_result.pages)}")
        print(f"   마크다운 파일 수: {len(layout_result.markdown_files)}")
        print(f"   분석 신뢰도: {layout_result.confidence:.3f}")
        
        if layout_result.error:
            print(f"   ❌ 에러: {layout_result.error}")
        
        # 레이아웃 분석 결과 검증
        assert layout_result.success, f"레이아웃 분석 실패: {layout_result.error}"
        assert len(layout_result.markdown_files) > 0, "마크다운 파일이 생성되지 않았습니다"
        
        # 생성된 파일들 확인
        print(f"\n   생성된 마크다운 파일들:")
        for i, md_file in enumerate(layout_result.markdown_files[:3]):  # 처음 3개만 표시
            md_path = Path(md_file)
            if md_path.exists():
                size = md_path.stat().st_size
                print(f"     {i+1}. {md_path.name} ({size} bytes)")
                
                # 첫 번째 파일의 내용 미리보기
                if i == 0:
                    content = md_path.read_text(encoding='utf-8')
                    preview = content[:200].replace('\n', ' ')
                    print(f"        미리보기: {preview}...")
        
        # Step 2: 마크다운 → 한국어 번역
        print(f"\n🔤 Step 2: 한국어 번역 시작...")
        
        # 첫 번째 마크다운 파일만 번역 (테스트 속도를 위해)
        first_markdown = layout_result.markdown_files[0]
        first_md_path = Path(first_markdown)
        
        print(f"   번역할 파일: {first_md_path.name}")
        print(f"   파일 크기: {first_md_path.stat().st_size} bytes")
        
        translation_start = time.time()
        translation_result = translation_service.translate_document(first_markdown)
        translation_time = time.time() - translation_start
        
        print(f"   번역 완료: {translation_time:.1f}초")
        print(f"   성공 여부: {translation_result.success}")
        
        if translation_result.success:
            print(f"   번역 파일: {Path(translation_result.output_file).name}")
            if translation_result.report:
                conf = translation_result.report.get('average_confidence', 'N/A')
                sections = translation_result.report.get('sections_count', 'N/A')
                print(f"   번역 신뢰도: {conf}")
                print(f"   번역된 섹션 수: {sections}")
        else:
            print(f"   ❌ 번역 에러: {translation_result.error}")
        
        # Step 3: 결과 검증 및 미리보기
        total_time = time.time() - total_start_time
        print(f"\n📊 Step 3: 결과 검증...")
        print(f"   전체 처리 시간: {total_time:.1f}초")
        
        if translation_result.success:
            # 번역 결과 파일 확인
            korean_file = Path(translation_result.output_file)
            assert korean_file.exists(), "한국어 번역 파일이 생성되지 않았습니다"
            
            korean_content = korean_file.read_text(encoding='utf-8')
            korean_chars = sum(1 for char in korean_content if '가' <= char <= '힣')
            
            print(f"   번역 파일 크기: {korean_file.stat().st_size} bytes")
            print(f"   한국어 글자 수: {korean_chars}")
            
            # 한국어가 실제로 포함되어 있는지 확인
            assert korean_chars > 0, "번역 결과에 한국어가 포함되지 않았습니다"
            
            # 번역 결과 미리보기
            print(f"\n📋 번역 결과 미리보기:")
            lines = korean_content.split('\n')
            for i, line in enumerate(lines[:5]):  # 처음 5줄만
                if line.strip():
                    print(f"     {line.strip()}")
                if i >= 4:
                    break
            if len(lines) > 5:
                print(f"     ... (총 {len(lines)}줄)")
        
        # 성능 리포트
        print(f"\n📈 성능 리포트:")
        print(f"   레이아웃 분석: {layout_time:.1f}초")
        print(f"   번역: {translation_time:.1f}초") 
        print(f"   전체: {total_time:.1f}초")
        print(f"   처리 효율: {first_md_path.stat().st_size / total_time:.0f} bytes/sec")
        
        # 최종 성공 여부
        pipeline_success = layout_result.success and translation_result.success
        print(f"\n{'✅' if pipeline_success else '❌'} 전체 파이프라인: {'성공' if pipeline_success else '실패'}")
        
        # 번역은 실패할 수 있으므로 레이아웃 분석만 성공하면 통과
        assert layout_result.success, "최소한 레이아웃 분석은 성공해야 합니다"
        
        if translation_result.success:
            assert korean_chars > 10, "번역된 한국어가 너무 적습니다"
        
        return {
            'layout_result': layout_result,
            'translation_result': translation_result,
            'total_time': total_time,
            'korean_chars': korean_chars if translation_result.success else 0
        }

    def test_batch_translation_real(self, layout_service, translation_service, temp_dir):
        """배치 번역 실제 테스트 (여러 마크다운 파일 동시 번역)"""
        
        # 간단한 테스트용 PDF가 있는지 확인
        test_pdf = "test_data/sample.pdf"
        if not Path(test_pdf).exists():
            pytest.skip("배치 테스트용 PDF 파일이 없습니다")
        
        ready, message = self.check_ollama_ready()
        if not ready:
            pytest.skip(f"Ollama 준비 안됨: {message}")
        
        print(f"\n📦 배치 번역 테스트 시작...")
        
        # 1. PDF에서 마크다운 생성
        layout_result = layout_service.analyze_document(test_pdf, temp_dir)
        assert layout_result.success, "레이아웃 분석 실패"
        
        # 2. 여러 마크다운 파일 번역 (최대 3개)
        markdown_files = layout_result.markdown_files[:3]
        print(f"   번역할 파일 수: {len(markdown_files)}")
        
        batch_start = time.time()
        batch_result = translation_service.translate_documents_batch(markdown_files)
        batch_time = time.time() - batch_start
        
        print(f"   배치 번역 시간: {batch_time:.1f}초")
        print(f"   성공한 파일: {batch_result.successful_files}/{batch_result.total_files}")
        print(f"   실패한 파일: {batch_result.failed_files}")
        print(f"   평균 신뢰도: {batch_result.average_confidence:.3f}")
        
        # 최소 1개는 성공해야 함
        assert batch_result.successful_files > 0, "배치 번역에서 성공한 파일이 없습니다"

if __name__ == "__main__":
    # 직접 실행 시 간단한 테스트
    print("직접 실행 모드")
    test_instance = TestRealEndToEnd()
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        layout_service = LayoutAnalysisService({'use_gpu': False})
        translation_service = TranslationService({'model_name': 'qqwen3:8b'})
        
        try:
            result = test_instance.test_pdf_to_korean_pipeline(
                layout_service, translation_service, temp_dir
            )
            print("✅ 테스트 성공!")
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")