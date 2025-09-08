# run_server.py
"""
문서 번역 서비스 서버 실행 스크립트
현재 프로젝트 구조에 맞춰 수정됨
"""
import sys
import logging
from pathlib import Path
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경변수 설정
os.environ.setdefault("PYTHONPATH", str(project_root))

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('translation_service.log')
        ]
    )


def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "uploads",
        "outputs", 
        "workflow_states",
        "logs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")


def check_dependencies():
    """기본 의존성 확인"""
    try:
        import fastapi
        import uvicorn
        print("✓ FastAPI dependencies OK")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # 기존 서비스 파일들 확인
    required_files = [
        "services/layout_analysis_service_paged.py",
        "services/translation_service.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing required file: {file_path}")
            return False
        else:
            print(f"✓ Found: {file_path}")
    
    return True


def main():
    """메인 실행 함수"""
    print("🚀 문서 번역 서비스를 시작합니다...")
    
    # 의존성 확인
    if not check_dependencies():
        print("❌ 의존성 확인 실패. 종료합니다.")
        return
    
    # 로깅 설정
    setup_logging()
    
    # 필요한 디렉토리 생성
    create_directories()
    
    # 서버 시작
    print("\n📡 서버 정보:")
    print("- URL: http://localhost:8000")
    print("- 웹 인터페이스: http://localhost:8000")
    print("- API 문서: http://localhost:8000/docs")
    print("- 헬스 체크: http://localhost:8000/health")
    print("\n🔧 개발 모드로 실행 중...")
    print("Ctrl+C로 종료할 수 있습니다.\n")
    
    try:
        import uvicorn
        # uvicorn 서버 실행
        import platform
        # 윈도우에서는 워커 1개 권장 (프로세스 종료 지연/신호 처리 이슈 완화)
        # 워커는 일단 고정 1 (윈도우 종료 지연/자원 중복 로드 방지)
        workers = 1
        # 다중 워커 사용 시 --reload는 비활성화 (윈도우 종료 지연 방지)
        reload_flag = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
        shutdown_timeout = int(os.getenv("UVICORN_SHUTDOWN_TIMEOUT", "5"))
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=reload_flag,
            log_level="info",
            reload_dirs=[str(project_root)] if reload_flag else None,
            workers=workers,
            timeout_graceful_shutdown=shutdown_timeout,
            timeout_keep_alive=5
        )
    except KeyboardInterrupt:
        print("\n👋 서버를 종료합니다.")
    except Exception as e:
        print(f"❌ 서버 실행 실패: {e}")
        print("\n디버깅을 위해 다음 명령어로 직접 실행해보세요:")
        print("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()