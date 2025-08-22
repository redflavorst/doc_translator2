# config.py
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """프로젝트 설정 관리"""
    
    # 기본 디렉토리
    PROJECT_ROOT = Path(__file__).parent
    OUTPUT_DIR = PROJECT_ROOT / "output"
    TEMP_DIR = PROJECT_ROOT / "temp"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # PaddleOCR 설정
    PADDLE_CONFIG = {
        'use_gpu': os.getenv('USE_GPU', 'true').lower() == 'true',
        'det_limit_side_len': int(os.getenv('DET_LIMIT_SIDE_LEN', '1920')),
        'use_table': os.getenv('USE_TABLE', 'true').lower() == 'true',
        'batch_size': int(os.getenv('BATCH_SIZE', '8')),
    }
    
    # 번역 설정
    TRANSLATION_CONFIG = {
        'model_name': os.getenv('OLLAMA_MODEL', 'qwen3:8b'),
        'temperature': float(os.getenv('TEMPERATURE', '0.1')),
        'max_retries': int(os.getenv('MAX_RETRIES', '3')),
        'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
        'quality_threshold': float(os.getenv('QUALITY_THRESHOLD', '0.6')),
    }
    
    # 로깅 설정
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': str(LOGS_DIR / 'app.log'),
                'mode': 'a',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    @classmethod
    def setup_directories(cls):
        """필요한 디렉토리 생성"""
        for dir_path in [cls.OUTPUT_DIR, cls.TEMP_DIR, cls.LOGS_DIR