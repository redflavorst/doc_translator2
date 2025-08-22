# core/error_handler.py
import time
import logging
from typing import Callable, Any, Optional, Dict, Type
from functools import wraps
from enum import Enum

from .models import WorkflowState, WorkflowStage
from .workflow_manager import WorkflowManager
from .config import WorkflowConfig

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "LOW"           # 재시도로 해결 가능
    MEDIUM = "MEDIUM"     # 수동 개입 필요할 수 있음
    HIGH = "HIGH"         # 즉시 중단 필요
    CRITICAL = "CRITICAL" # 시스템 전체 영향


class RetryableError(Exception):
    """재시도 가능한 에러"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.LOW):
        super().__init__(message)
        self.severity = severity


class NonRetryableError(Exception):
    """재시도 불가능한 에러"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH):
        super().__init__(message)
        self.severity = severity


class WorkflowErrorHandler:
    """워크플로우 에러 처리 및 복구 시스템"""
    
    def __init__(self, workflow_manager: WorkflowManager, config: WorkflowConfig):
        """
        에러 핸들러 초기화
        
        Args:
            workflow_manager: 워크플로우 매니저
            config: 워크플로우 설정
        """
        self.workflow_manager = workflow_manager
        self.config = config
        self.error_statistics: Dict[str, int] = {}
    
    def with_error_handling(self, workflow_id: str, stage: WorkflowStage):
        """
        에러 처리 데코레이터
        
        Args:
            workflow_id: 워크플로우 ID
            stage: 현재 실행 단계
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                return self._execute_with_retry(func, workflow_id, stage, *args, **kwargs)
            return wrapper
        return decorator
    
    def _execute_with_retry(self, func: Callable, workflow_id: str, 
                           stage: WorkflowStage, *args, **kwargs) -> Any:
        """
        재시도 로직과 함께 함수 실행
        
        Args:
            func: 실행할 함수
            workflow_id: 워크플로우 ID
            stage: 현재 단계
            *args, **kwargs: 함수 인자
            
        Returns:
            함수 실행 결과
            
        Raises:
            NonRetryableError: 재시도 불가능한 에러 발생 시
        """
        retry_count = 0
        last_exception = None
        
        while retry_count <= self.config.max_retry_count:
            try:
                # 함수 실행
                result = func(*args, **kwargs)
                
                # 성공 시 에러 정보 초기화
                if retry_count > 0:
                    logger.info(f"Workflow {workflow_id} recovered after {retry_count} retries")
                
                return result
                
            except NonRetryableError as e:
                # 재시도 불가능한 에러
                self._handle_non_retryable_error(workflow_id, stage, e)
                raise
                
            except RetryableError as e:
                # 재시도 가능한 에러
                last_exception = e
                retry_count += 1
                
                if retry_count > self.config.max_retry_count:
                    break
                
                logger.warning(
                    f"Workflow {workflow_id} stage {stage.value} failed "
                    f"(attempt {retry_count}/{self.config.max_retry_count}): {e}"
                )
                
                # 재시도 전 대기
                time.sleep(self.config.retry_delay_seconds * retry_count)
                
            except Exception as e:
                # 일반 예외는 재시도 가능한 에러로 처리
                last_exception = RetryableError(str(e))
                retry_count += 1
                
                if retry_count > self.config.max_retry_count:
                    break
                
                logger.warning(
                    f"Workflow {workflow_id} stage {stage.value} failed "
                    f"(attempt {retry_count}/{self.config.max_retry_count}): {e}"
                )
                
                time.sleep(self.config.retry_delay_seconds * retry_count)
        
        # 모든 재시도 실패
        self._handle_retry_exhausted(workflow_id, stage, last_exception)
        raise last_exception
    
    def _handle_non_retryable_error(self, workflow_id: str, stage: WorkflowStage, 
                                   error: NonRetryableError) -> None:
        """
        재시도 불가능한 에러 처리
        
        Args:
            workflow_id: 워크플로우 ID
            stage: 실패한 단계
            error: 발생한 에러
        """
        error_type = type(error).__name__
        
        # 워크플로우에 에러 정보 저장
        self.workflow_manager.set_workflow_error(
            workflow_id=workflow_id,
            error_type=error_type,
            message=str(error),
            traceback=self._get_traceback()
        )
        
        # 에러 통계 업데이트
        self._update_error_statistics(error_type)
        
        logger.error(
            f"Non-retryable error in workflow {workflow_id} stage {stage.value}: "
            f"{error_type} - {error}"
        )
    
    def _handle_retry_exhausted(self, workflow_id: str, stage: WorkflowStage, 
                               last_error: Exception) -> None:
        """
        재시도 횟수 초과 시 처리
        
        Args:
            workflow_id: 워크플로우 ID
            stage: 실패한 단계
            last_error: 마지막 에러
        """
        error_type = type(last_error).__name__
        
        # 워크플로우에 에러 정보 저장
        self.workflow_manager.set_workflow_error(
            workflow_id=workflow_id,
            error_type=error_type,
            message=f"Max retry count exceeded: {last_error}",
            traceback=self._get_traceback()
        )
        
        # 에러 통계 업데이트
        self._update_error_statistics(f"{error_type}_retry_exhausted")
        
        logger.error(
            f"Retry exhausted for workflow {workflow_id} stage {stage.value}: "
            f"{error_type} - {last_error}"
        )
    
    def _update_error_statistics(self, error_type: str) -> None:
        """
        에러 통계 업데이트
        
        Args:
            error_type: 에러 타입
        """
        self.error_statistics[error_type] = self.error_statistics.get(error_type, 0) + 1
    
    def _get_traceback(self) -> str:
        """
        현재 스택 트레이스 반환
        
        Returns:
            스택 트레이스 문자열
        """
        import traceback
        return traceback.format_exc()
    
    def get_error_statistics(self) -> Dict[str, int]:
        """
        에러 통계 조회
        
        Returns:
            에러 타입별 발생 횟수
        """
        return self.error_statistics.copy()
    
    def reset_error_statistics(self) -> None:
        """에러 통계 초기화"""
        self.error_statistics.clear()
        logger.info("Error statistics reset")
    
    def recover_failed_workflows(self) -> int:
        """
        실패한 워크플로우들을 자동 복구 시도
        
        Returns:
            복구된 워크플로우 수
        """
        recoverable_workflows = self.workflow_manager.get_recoverable_workflows()
        recovered_count = 0
        
        for workflow in recoverable_workflows:
            try:
                # 복구 조건 확인
                if self._should_auto_recover(workflow):
                    self.workflow_manager.recover_workflow(workflow.id)
                    recovered_count += 1
                    logger.info(f"Auto-recovered workflow: {workflow.id}")
            except Exception as e:
                logger.error(f"Failed to auto-recover workflow {workflow.id}: {e}")
        
        return recovered_count
    
    def _should_auto_recover(self, workflow: WorkflowState) -> bool:
        """
        자동 복구 가능 여부 판단
        
        Args:
            workflow: 워크플로우 상태
            
        Returns:
            자동 복구 가능 여부
        """
        # 에러 정보가 없으면 복구 불가
        if not workflow.error_info:
            return False
        
        # 특정 에러 타입은 자동 복구 불가
        non_recoverable_errors = [
            "FileNotFoundError",
            "PermissionError",
            "ValidationError",
            "NonRetryableError"
        ]
        
        error_type = workflow.error_info.get("error_type", "")
        if error_type in non_recoverable_errors:
            return False
        
        # 실패 후 일정 시간 경과 확인 (쿨다운)
        if workflow.updated_at:
            cooldown_minutes = 10
            from datetime import datetime, timedelta
            cooldown_time = workflow.updated_at + timedelta(minutes=cooldown_minutes)
            if datetime.now() < cooldown_time:
                return False
        
        return True


# 특정 에러 타입들
class OCRError(RetryableError):
    """OCR 처리 에러"""
    pass


class TranslationError(RetryableError):
    """번역 처리 에러"""
    pass


class FileProcessingError(NonRetryableError):
    """파일 처리 에러"""
    pass


class ValidationError(NonRetryableError):
    """유효성 검증 에러"""
    pass


class ResourceExhaustionError(RetryableError):
    """리소스 부족 에러"""
    def __init__(self, message: str):
        super().__init__(message, ErrorSeverity.MEDIUM)