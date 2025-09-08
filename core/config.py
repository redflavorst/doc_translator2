# core/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
try:
    import paddle  # GPU 기본값 판단용 (선택)
    _PADDLE_CUDA_COMPILED = bool(getattr(paddle, 'is_compiled_with_cuda', lambda: False)())
except Exception:
    _PADDLE_CUDA_COMPILED = False


@dataclass
class WorkflowConfig:
    """워크플로우 관련 설정"""
    state_directory: str = "./workflow_states"
    max_retry_count: int = 3
    retry_delay_seconds: int = 5
    auto_cleanup_days: int = 7
    max_concurrent_workflows: int = 5


@dataclass
class LayoutAnalysisConfig:
    """레이아웃 분석 설정"""
    paddle_model_dir: Optional[str] = None
    use_gpu: bool = True
    batch_size: int = 1
    confidence_threshold: float = 0.5
    # HPI/HPIP 관련 설정
    use_hpip: bool = False
    hpi_backend: str = "tensorrt"  # tensorrt | onnxruntime
    hpi_device: str = "gpu"        # gpu | cpu


@dataclass
class TranslationConfig:
    """번역 설정"""
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "gemma3n:e4b"
    temperature: float = 0.1
    max_tokens: int = 2048
    quality_threshold: float = 0.6


@dataclass
class AppConfig:
    """애플리케이션 전체 설정"""
    workflow: WorkflowConfig
    layout_analysis: LayoutAnalysisConfig
    translation: TranslationConfig
    
    # 로깅 설정
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # API 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    @classmethod
    def from_environment(cls) -> 'AppConfig':
        """환경변수에서 설정 로드"""
        return cls(
            workflow=WorkflowConfig(
                state_directory=os.getenv("WORKFLOW_STATE_DIR", "./workflow_states"),
                max_retry_count=int(os.getenv("MAX_RETRY_COUNT", "3")),
                retry_delay_seconds=int(os.getenv("RETRY_DELAY_SECONDS", "5")),
                auto_cleanup_days=int(os.getenv("AUTO_CLEANUP_DAYS", "7")),
                max_concurrent_workflows=int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "5"))
            ),
            layout_analysis=LayoutAnalysisConfig(
                paddle_model_dir=os.getenv("PADDLE_MODEL_DIR"),
                # USE_GPU 명시 시 그 값을 사용, 미설정 시 CUDA 빌드면 기본 True
                use_gpu=(os.getenv("USE_GPU").lower() == "true") if os.getenv("USE_GPU") is not None else _PADDLE_CUDA_COMPILED,
                batch_size=int(os.getenv("BATCH_SIZE", "1")),
                confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
                use_hpip=os.getenv("USE_HPIP", "false").lower() == "true",
                hpi_backend=os.getenv("HPI_BACKEND", "tensorrt"),
                hpi_device=os.getenv("HPI_DEVICE", "gpu")
            ),
            translation=TranslationConfig(
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model_name=os.getenv("TRANSLATION_MODEL", "gemma3n:e4b"),
                temperature=float(os.getenv("TRANSLATION_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
                quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.6"))
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            api_debug=os.getenv("API_DEBUG", "false").lower() == "true"
        )
    
    def validate(self) -> None:
        """설정 유효성 검증"""
        # 디렉토리 존재 확인 및 생성
        state_dir = Path(self.workflow.state_directory)
        state_dir.mkdir(exist_ok=True)
        
        # 값 범위 검증
        if self.workflow.max_retry_count < 0:
            raise ValueError("max_retry_count must be >= 0")
        
        if self.workflow.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be >= 0")
        
        if self.layout_analysis.confidence_threshold < 0 or self.layout_analysis.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.translation.quality_threshold < 0 or self.translation.quality_threshold > 1:
            raise ValueError("quality_threshold must be between 0 and 1")


# 글로벌 설정 인스턴스
app_config = AppConfig.from_environment()