# core/workflow_manager.py
import uuid
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from .models import WorkflowState, WorkflowStatus, WorkflowStage
from .config import WorkflowConfig

logger = logging.getLogger(__name__)


class WorkflowManager:
    """워크플로우 상태를 관리하는 매니저 클래스"""
    
    def __init__(self, config: WorkflowConfig):
        """
        WorkflowManager 초기화
        
        Args:
            config: 워크플로우 설정
        """
        self.config = config
        self.state_dir = Path(config.state_directory)
        self.state_dir.mkdir(exist_ok=True)
        
        logger.info(f"WorkflowManager initialized with state directory: {self.state_dir}")
    
    def create_workflow(self, input_file_path: str, output_directory: str) -> str:
        """
        새로운 워크플로우 생성
        
        Args:
            input_file_path: 입력 PDF 파일 경로
            output_directory: 출력 디렉토리 경로
            
        Returns:
            생성된 워크플로우 ID
            
        Raises:
            RuntimeError: 동시 실행 제한 초과 시
        """
        # 동시 실행 제한 확인
        self._check_concurrent_limit()
        
        # 새 워크플로우 ID 생성
        workflow_id = str(uuid.uuid4())
        
        # 워크플로우 상태 생성
        state = WorkflowState(
            id=workflow_id,
            status=WorkflowStatus.CREATED,
            current_stage=WorkflowStage.LAYOUT_ANALYSIS,
            stages_completed=[],
            input_file_path=input_file_path,
            output_directory=output_directory
        )
        
        # 유효성 검증
        state.validate()
        
        # 상태 저장
        self._save_state(state)
        
        logger.info(f"Created workflow {workflow_id} for file: {input_file_path}")
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> WorkflowState:
        """
        워크플로우 상태 조회
        
        Args:
            workflow_id: 워크플로우 ID
            
        Returns:
            워크플로우 상태
            
        Raises:
            FileNotFoundError: 워크플로우를 찾을 수 없는 경우
        """
        state_file = self.state_dir / f"{workflow_id}.json"
        
        if not state_file.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_id}")
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return WorkflowState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load workflow {workflow_id}: {e}")
            raise ValueError(f"Invalid workflow state file: {workflow_id}")
    
    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus) -> None:
        """
        워크플로우 상태 업데이트
        
        Args:
            workflow_id: 워크플로우 ID
            status: 새로운 상태
        """
        state = self.get_workflow(workflow_id)
        
        # 동시 실행 제한 확인 (RUNNING으로 변경 시)
        if status == WorkflowStatus.RUNNING and state.status != WorkflowStatus.RUNNING:
            self._check_concurrent_limit(exclude_workflow_id=workflow_id)
        
        old_status = state.status
        state.status = status
        state.updated_at = datetime.now()
        
        # 상태별 특별 처리
        if status == WorkflowStatus.RUNNING:
            state.start()
        
        self._save_state(state)
        logger.info(f"Workflow {workflow_id} status changed: {old_status.value} -> {status.value}")
    
    def update_progress(self, workflow_id: str, current_page: int = None, 
                       total_pages: int = None, current_action: str = None,
                       stage_details: Dict[str, Any] = None,
                       pages_analyzed: int = None, pages_translated: int = None):
        """
        워크플로우 진행 상황 업데이트
        
        Args:
            workflow_id: 워크플로우 ID
            current_page: 현재 처리 중인 페이지
            total_pages: 전체 페이지 수
            current_action: 현재 수행 중인 작업 설명
            stage_details: 단계별 상세 정보
            pages_analyzed: 레이아웃 분석 완료된 페이지 수
            pages_translated: 번역 완료된 페이지 수
        """
        state = self.get_workflow(workflow_id)
        
        if current_page is not None:
            state.current_page = current_page
        if total_pages is not None:
            state.total_pages = total_pages
        if current_action is not None:
            state.current_action = current_action
        if stage_details is not None:
            state.stage_details = stage_details
        if pages_analyzed is not None:
            state.pages_analyzed = pages_analyzed
        if pages_translated is not None:
            state.pages_translated = pages_translated
        
        # 진행률 재계산
        state._update_progress()
        
        state.updated_at = datetime.now()
        self._save_state(state)
        logger.debug(f"Workflow {workflow_id} progress updated: Page {current_page}/{total_pages} - {current_action}")
    
    def complete_stage(self, workflow_id: str, stage: WorkflowStage) -> None:
        """
        워크플로우 단계 완료 처리
        
        Args:
            workflow_id: 워크플로우 ID
            stage: 완료된 단계
        """
        state = self.get_workflow(workflow_id)
        state.complete_stage(stage)
        self._save_state(state)
        
        logger.info(f"Workflow {workflow_id} completed stage: {stage.value}")
    
    def set_workflow_error(self, workflow_id: str, error_type: str, 
                          message: str, traceback: str = None) -> None:
        """
        워크플로우 에러 설정
        
        Args:
            workflow_id: 워크플로우 ID
            error_type: 에러 타입
            message: 에러 메시지
            traceback: 스택 트레이스
        """
        state = self.get_workflow(workflow_id)
        state.set_error(error_type, message, traceback)
        self._save_state(state)
        
        logger.error(f"Workflow {workflow_id} failed: {error_type} - {message}")
    
    def list_workflows_by_status(self, status: WorkflowStatus) -> List[WorkflowState]:
        """
        상태별 워크플로우 목록 조회
        
        Args:
            status: 조회할 상태
            
        Returns:
            해당 상태의 워크플로우 목록
        """
        workflows = []
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                state = self._load_state_from_file(state_file)
                if state.status == status:
                    workflows.append(state)
            except Exception as e:
                logger.warning(f"Failed to load workflow from {state_file}: {e}")
        
        # 생성일자 순 정렬
        workflows.sort(key=lambda w: w.created_at)
        return workflows
    
    def get_recoverable_workflows(self) -> List[WorkflowState]:
        """
        복구 가능한 워크플로우 목록 조회
        
        Returns:
            복구 가능한 워크플로우 목록
        """
        recoverable = []
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                state = self._load_state_from_file(state_file)
                if state.is_recoverable():
                    recoverable.append(state)
            except Exception as e:
                logger.warning(f"Failed to load workflow from {state_file}: {e}")
        
        return recoverable
    
    def recover_workflow(self, workflow_id: str) -> WorkflowState:
        """
        실패한 워크플로우 복구
        
        Args:
            workflow_id: 복구할 워크플로우 ID
            
        Returns:
            복구된 워크플로우 상태
            
        Raises:
            ValueError: 복구 불가능한 상태인 경우
        """
        state = self.get_workflow(workflow_id)
        
        if not state.is_recoverable():
            raise ValueError(f"Workflow is not in a recoverable state: {state.status.value}")
        
        # 동시 실행 제한 확인
        self._check_concurrent_limit(exclude_workflow_id=workflow_id)
        
        # 상태 복구
        state.status = WorkflowStatus.RUNNING
        state.error_info = None
        state.updated_at = datetime.now()
        
        self._save_state(state)
        logger.info(f"Recovered workflow {workflow_id}")
        
        return state
    
    def delete_workflow(self, workflow_id: str) -> None:
        """
        워크플로우 삭제
        
        Args:
            workflow_id: 삭제할 워크플로우 ID
        """
        state_file = self.state_dir / f"{workflow_id}.json"
        
        if state_file.exists():
            state_file.unlink()
            logger.info(f"Deleted workflow {workflow_id}")
        else:
            logger.warning(f"Workflow file not found for deletion: {workflow_id}")
    
    def cleanup_old_workflows(self) -> int:
        """
        오래된 워크플로우 정리
        
        Returns:
            삭제된 워크플로우 수
        """
        cutoff_date = datetime.now() - timedelta(days=self.config.auto_cleanup_days)
        deleted_count = 0
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                state = self._load_state_from_file(state_file)
                if state.created_at < cutoff_date and state.status in [
                    WorkflowStatus.COMPLETED, WorkflowStatus.FAILED
                ]:
                    state_file.unlink()
                    deleted_count += 1
                    logger.info(f"Cleaned up old workflow: {state.id}")
            except Exception as e:
                logger.warning(f"Failed to process workflow file {state_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old workflows")
        return deleted_count
    
    def get_workflow_statistics(self) -> Dict[str, int]:
        """
        워크플로우 통계 조회
        
        Returns:
            상태별 워크플로우 수
        """
        stats = {
            "total": 0,
            "created": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "paused": 0
        }
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                state = self._load_state_from_file(state_file)
                stats["total"] += 1
                stats[state.status.value.lower()] += 1
            except Exception as e:
                logger.warning(f"Failed to load workflow from {state_file}: {e}")
        
        return stats
    
    def pause_workflow(self, workflow_id: str) -> None:
        """
        워크플로우 일시 정지
        
        Args:
            workflow_id: 일시 정지할 워크플로우 ID
        """
        state = self.get_workflow(workflow_id)
        state.pause()
        self._save_state(state)
        logger.info(f"Paused workflow {workflow_id}")
    
    def resume_workflow(self, workflow_id: str) -> None:
        """
        워크플로우 재시작
        
        Args:
            workflow_id: 재시작할 워크플로우 ID
        """
        state = self.get_workflow(workflow_id)
        
        # 동시 실행 제한 확인
        self._check_concurrent_limit(exclude_workflow_id=workflow_id)
        
        state.resume()
        self._save_state(state)
        logger.info(f"Resumed workflow {workflow_id}")
    
    def _save_state(self, state: WorkflowState) -> None:
        """
        워크플로우 상태를 파일에 저장
        
        Args:
            state: 저장할 워크플로우 상태
        """
        state_file = self.state_dir / f"{state.id}.json"
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save workflow state {state.id}: {e}")
            raise
    
    def _load_state_from_file(self, state_file: Path) -> WorkflowState:
        """
        파일에서 워크플로우 상태 로드
        
        Args:
            state_file: 상태 파일 경로
            
        Returns:
            로드된 워크플로우 상태
        """
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return WorkflowState.from_dict(data)
    
    def _check_concurrent_limit(self, exclude_workflow_id: str = None) -> None:
        """
        동시 실행 워크플로우 제한 확인
        
        Args:
            exclude_workflow_id: 제외할 워크플로우 ID (상태 변경 시)
            
        Raises:
            RuntimeError: 제한 초과 시
        """
        running_workflows = self.list_workflows_by_status(WorkflowStatus.RUNNING)
        
        # 제외할 워크플로우가 있다면 필터링
        if exclude_workflow_id:
            running_workflows = [w for w in running_workflows if w.id != exclude_workflow_id]
        
        if len(running_workflows) >= self.config.max_concurrent_workflows:
            raise RuntimeError(
                f"Maximum concurrent workflows limit reached: "
                f"{self.config.max_concurrent_workflows}"
            )
    
    def get_running_workflows_count(self) -> int:
        """
        현재 실행 중인 워크플로우 수 조회
        
        Returns:
            실행 중인 워크플로우 수
        """
        return len(self.list_workflows_by_status(WorkflowStatus.RUNNING))