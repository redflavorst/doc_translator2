# tests/conftest.py
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 현재 작업 디렉토리도 프로젝트 루트로 변경
os.chdir(project_root)

# 환경 변수 설정
os.environ['PYTHONPATH'] = str(project_root)

# 디버깅을 위한 정보 출력
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")  # 처음 3개만 출력
print(f"Current working directory: {os.getcwd()}")

# services 모듈 로드 테스트
try:
    import services
    print("✓ services 모듈 로드 성공")
except ImportError as e:
    print(f"✗ services 모듈 로드 실패: {e}")

# pytest 설정
import pytest

def pytest_configure(config):
    """pytest 설정"""
    import logging
    logging.basicConfig(level=logging.INFO)

def pytest_collection_modifyitems(config, items):
    """테스트 아이템 수정"""
    # 통합 테스트에 마커 추가
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "real" in item.nodeid:
            item.add_marker(pytest.mark.real)