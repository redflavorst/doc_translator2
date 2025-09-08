# scripts/test_pipeline.py
"""
간단한 PDF → 한국어 번역 파이프라인 테스트 스크립트

사용법:
python scripts/test_pipeline.py [PDF파일경로]
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import tempfile
import shutil
from services.translation_service import TranslationService

def test_pipeline(pdf_path: str = None):
    """간단한 파이프라인 테스트"""
    
    # PDF 파일 경로 결정
    if pdf_path:
        pdf_path = Path(pdf_path)
    else:
        # 기본 테스트 파일들
        test_files = [
            "이전가격_0804/특수관계자계약서_FY2023_합본.pdf",
        ]
        pdf_path = None
        for file in test_files:
            if Path(file).exists():
                pdf_path = Path(file)
                break
    
    if not pdf_path or not pdf_path.exists():
        print("❌ PDF 파일을 찾을 수 없습니다.")
        print("사용법: python scripts/test_pipeline.py [PDF파일경로]")
        print("또는 test_data/sample.pdf 파일을 준비하세요.")
        return False
    
    print(f"🚀 파이프라인 테스트 시작")
    print(f"📄 PDF 파일: {pdf_path}")
    print("-" * 50)
    
    # 임시 출력 디렉토리
    output_dir = Path(f"./pipeline_test_output_{int(time.time())}")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 서비스 초기화
        layout_service = LayoutAnalysisService({
            'use_gpu': True,
            'det_limit_side_len': 1920,
            'use_table': True
        })
        
        translation_service = TranslationService({
            'model_name': 'qwen2.5:0.5b',
            'temperature': 0.1,
            'max_retries': 1,
            'quality_threshold': 0.3
        })
        
        # Step 1: 레이아웃 분석
        print("📊 Step 1: PDF 레이아웃 분석...")
        start_time = time.time()
        
        layout_result = layout_service.analyze_document(str(pdf_path), str(output_dir))
        
        if not layout_result.success:
            print(f"❌ 레이아웃 분석 실패: {layout_result.error}")
            return False
        
        layout_time = time.time() - start_time
        print(f"✅ 레이아웃 분석 완료 ({layout_time:.1f}초)")
        print(f"   📑 생성된 마크다운 파일: {len(layout_result.markdown_files)}개")
        print(f"   📊 분석 신뢰도: {layout_result.confidence:.3f}")
        
        # 첫 번째 마크다운 파일 정보
        if layout_result.markdown_files:
            first_md = Path(layout_result.markdown_files[0])
            print(f"   📄 첫 번째 파일: {first_md.name} ({first_md.stat().st_size} bytes)")
            
            # 내용 미리보기
            content = first_md.read_text(encoding='utf-8')
            preview = content[:150].replace('\n', ' ')
            print(f"   👀 내용 미리보기: {preview}...")
        
        # Step 2: 번역 (첫 번째 파일만)
        print(f"\n🔤 Step 2: 한국어 번역...")
        
        # Ollama 서버 확인
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=3)
            if response.status_code != 200:
                print("⚠️  Ollama 서버가 실행되지 않았습니다.")
                print("   번역을 건너뛰고 레이아웃 분석 결과만 확인합니다.")
                print(f"   Ollama 시작 명령: ollama serve")
                return True
        except:
            print("⚠️  Ollama 서버 연결 실패. 번역을 건너뜁니다.")
            return True
        
        if layout_result.markdown_files:
            first_markdown = layout_result.markdown_files[0]
            
            start_time = time.time()
            translation_result = translation_service.translate_document(first_markdown)
            translation_time = time.time() - start_time
            
            if translation_result.success:
                print(f"✅ 번역 완료 ({translation_time:.1f}초)")
                print(f"   📄 번역 파일: {Path(translation_result.output_file).name}")
                
                if translation_result.report:
                    conf = translation_result.report.get('average_confidence', 'N/A')
                    print(f"   📊 번역 신뢰도: {conf}")
                
                # 번역 결과 미리보기
                korean_file = Path(translation_result.output_file)
                korean_content = korean_file.read_text(encoding='utf-8')
                korean_chars = sum(1 for char in korean_content if '가' <= char <= '힣')
                
                print(f"   🇰🇷 한국어 글자 수: {korean_chars}")
                
                # 번역 결과 일부 출력
                lines = korean_content.split('\n')
                print(f"   👀 번역 결과 미리보기:")
                for line in lines[:3]:
                    if line.strip():
                        print(f"      {line.strip()}")
                        break
                
            else:
                print(f"❌ 번역 실패: {translation_result.error}")
                print("   레이아웃 분석은 성공했으므로 부분적 성공입니다.")
        
        print(f"\n📁 결과 파일들이 저장된 위치: {output_dir}")
        print("🎉 파이프라인 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 임시 디렉토리는 삭제하지 않고 유지 (결과 확인용)
        print(f"💡 결과 파일을 확인한 후 다음 명령으로 정리하세요:")
        print(f"   rmdir /s \"{output_dir}\"")

def main():
    """메인 함수"""
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_pipeline(pdf_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()