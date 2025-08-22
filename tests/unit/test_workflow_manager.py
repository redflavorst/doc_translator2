# tests/unit/test_workflow_manager.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from core.workflow_manager import WorkflowManager
from core.models import WorkflowState, WorkflowStatus, WorkflowStage
from core.config import WorkflowConfig


class TestWorkflowManager:
    """WorkflowManager 테스트"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.config = WorkflowConfig(
            state_directory=self.temp_dir,
            max_retry_count=3,
            retry_delay_seconds=1,
            auto_cleanup_days=7
        )
        self.manager = WorkflowManager(self.config)
    
    def teardown_method(self):
        """각 테스트 후 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir)
    
    def test_create_workflow(self):
        """워크플로우 생성 테스트"""
        # Given
        input_file = "/path/to/test.pdf"
        output_dir = "/path/to/output"
        
        # When
        workflow_id = self.manager.create_workflow(input_file, output_dir)
        
        # Then
        assert workflow_id is not None
        assert len(workflow_id) > 0
        
        # 상태 파일이 생성되었는지 확인
        state_file = Path(self.temp_dir) / f"{workflow_id}.json"
        assert state_file.exists()
        
        # 워크플로우 상태 확인
        state = self.manager.get_workflow(workflow_id)
        assert state.id == workflow_id
        assert state.status == WorkflowStatus.CREATED
        assert state.input_file_path == input_file
        assert state.output_directory == output_dir
    
    def test_get_workflow_existing(self):
        """기존 워크플로우 조회 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        
        # When
        state = self.manager.get_workflow(workflow_id)
        
        # Then
        assert state is not None
        assert state.id == workflow_id
        assert state.status == WorkflowStatus.CREATED
    
    def test_get_workflow_not_found(self):
        """존재하지 않는 워크플로우 조회 테스트"""
        # Given
        non_existent_id = "non-existent-id"
        
        # When & Then
        with pytest.raises(FileNotFoundError, match="Workflow not found"):
            self.manager.get_workflow(non_existent_id)
    
    def test_update_workflow_status(self):
        """워크플로우 상태 업데이트 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        
        # When
        self.manager.update_workflow_status(workflow_id, WorkflowStatus.RUNNING)
        
        # Then
        state = self.manager.get_workflow(workflow_id)
        assert state.status == WorkflowStatus.RUNNING
    
    def test_complete_stage(self):
        """단계 완료 처리 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        self.manager.update_workflow_status(workflow_id, WorkflowStatus.RUNNING)
        
        # When
        self.manager.complete_stage(workflow_id, WorkflowStage.LAYOUT_ANALYSIS)
        
        # Then
        state = self.manager.get_workflow(workflow_id)
        assert WorkflowStage.LAYOUT_ANALYSIS.value in state.stages_completed
        assert state.current_stage == WorkflowStage.TRANSLATION
        assert state.progress_percentage == 33  # 3단계 중 1단계 완료
    
    def test_set_workflow_error(self):
        """워크플로우 에러 설정 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        error_type = "OCRError"
        error_message = "Failed to process PDF"
        traceback = "Traceback..."
        
        # When
        self.manager.set_workflow_error(workflow_id, error_type, error_message, traceback)
        
        # Then
        state = self.manager.get_workflow(workflow_id)
        assert state.status == WorkflowStatus.FAILED
        assert state.error_info["error_type"] == error_type
        assert state.error_info["message"] == error_message
        assert state.error_info["traceback"] == traceback
    
    def test_list_workflows_by_status(self):
        """상태별 워크플로우 목록 조회 테스트"""
        # Given
        id1 = self.manager.create_workflow("/test1.pdf", "/output1")
        id2 = self.manager.create_workflow("/test2.pdf", "/output2")
        id3 = self.manager.create_workflow("/test3.pdf", "/output3")
        
        self.manager.update_workflow_status(id1, WorkflowStatus.RUNNING)
        self.manager.update_workflow_status(id2, WorkflowStatus.RUNNING)
        self.manager.update_workflow_status(id3, WorkflowStatus.COMPLETED)
        
        # When
        running_workflows = self.manager.list_workflows_by_status(WorkflowStatus.RUNNING)
        completed_workflows = self.manager.list_workflows_by_status(WorkflowStatus.COMPLETED)
        
        # Then
        assert len(running_workflows) == 2
        assert len(completed_workflows) == 1
        assert id1 in [w.id for w in running_workflows]
        assert id2 in [w.id for w in running_workflows]
        assert id3 in [w.id for w in completed_workflows]
    
    def test_get_recoverable_workflows(self):
        """복구 가능한 워크플로우 조회 테스트"""
        # Given
        id1 = self.manager.create_workflow("/test1.pdf", "/output1")
        id2 = self.manager.create_workflow("/test2.pdf", "/output2")
        id3 = self.manager.create_workflow("/test3.pdf", "/output3")
        
        self.manager.update_workflow_status(id1, WorkflowStatus.FAILED)
        self.manager.update_workflow_status(id2, WorkflowStatus.PAUSED)
        self.manager.update_workflow_status(id3, WorkflowStatus.COMPLETED)
        
        # When
        recoverable = self.manager.get_recoverable_workflows()
        
        # Then
        assert len(recoverable) == 2
        recoverable_ids = [w.id for w in recoverable]
        assert id1 in recoverable_ids
        assert id2 in recoverable_ids
        assert id3 not in recoverable_ids
    
    def test_recover_workflow(self):
        """워크플로우 복구 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        self.manager.set_workflow_error(workflow_id, "TestError", "Test error message")
        
        # When
        recovered_state = self.manager.recover_workflow(workflow_id)
        
        # Then
        assert recovered_state.status == WorkflowStatus.RUNNING
        assert recovered_state.error_info is None
    
    def test_recover_non_recoverable_workflow(self):
        """복구 불가능한 워크플로우 복구 시도 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        self.manager.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED)
        
        # When & Then
        with pytest.raises(ValueError, match="Workflow is not in a recoverable state"):
            self.manager.recover_workflow(workflow_id)
    
    def test_delete_workflow(self):
        """워크플로우 삭제 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        
        # When
        self.manager.delete_workflow(workflow_id)
        
        # Then
        state_file = Path(self.temp_dir) / f"{workflow_id}.json"
        assert not state_file.exists()
        
        with pytest.raises(FileNotFoundError):
            self.manager.get_workflow(workflow_id)
    
    def test_cleanup_old_workflows(self):
        """오래된 워크플로우 정리 테스트"""
        # Given
        old_workflow_id = self.manager.create_workflow("/old.pdf", "/output")
        recent_workflow_id = self.manager.create_workflow("/recent.pdf", "/output")
        
        # 오래된 워크플로우를 완료 상태로 변경하고 날짜를 과거로 설정
        old_state = self.manager.get_workflow(old_workflow_id)
        old_state.status = WorkflowStatus.COMPLETED  # 완료 상태로 변경
        old_state.created_at = datetime.now() - timedelta(days=10)
        self.manager._save_state(old_state)
        
        # When
        deleted_count = self.manager.cleanup_old_workflows()
        
        # Then
        assert deleted_count == 1
        
        # 오래된 워크플로우는 삭제되고, 최근 워크플로우는 남아있어야 함
        with pytest.raises(FileNotFoundError):
            self.manager.get_workflow(old_workflow_id)
        
        recent_state = self.manager.get_workflow(recent_workflow_id)
        assert recent_state is not None
    
    def test_get_workflow_statistics(self):
        """워크플로우 통계 조회 테스트"""
        # Given
        id1 = self.manager.create_workflow("/test1.pdf", "/output1")
        id2 = self.manager.create_workflow("/test2.pdf", "/output2")
        id3 = self.manager.create_workflow("/test3.pdf", "/output3")
        
        self.manager.update_workflow_status(id1, WorkflowStatus.RUNNING)
        self.manager.update_workflow_status(id2, WorkflowStatus.COMPLETED)
        self.manager.update_workflow_status(id3, WorkflowStatus.FAILED)
        
        # When
        stats = self.manager.get_workflow_statistics()
        
        # Then
        assert stats["total"] == 3
        assert stats["created"] == 0
        assert stats["running"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["paused"] == 0
    
    def test_concurrent_workflow_limit(self):
        """동시 실행 워크플로우 제한 테스트"""
        # Given
        # config의 max_concurrent_workflows를 2로 설정
        self.config.max_concurrent_workflows = 2
        
        id1 = self.manager.create_workflow("/test1.pdf", "/output1")
        id2 = self.manager.create_workflow("/test2.pdf", "/output2")
        
        self.manager.update_workflow_status(id1, WorkflowStatus.RUNNING)
        self.manager.update_workflow_status(id2, WorkflowStatus.RUNNING)
        
        # When & Then
        with pytest.raises(RuntimeError, match="Maximum concurrent workflows limit reached"):
            id3 = self.manager.create_workflow("/test3.pdf", "/output3")
            self.manager.update_workflow_status(id3, WorkflowStatus.RUNNING)
    
    def test_save_and_load_state_persistence(self):
        """상태 저장/로드 지속성 테스트"""
        # Given
        workflow_id = self.manager.create_workflow("/test.pdf", "/output")
        self.manager.complete_stage(workflow_id, WorkflowStage.LAYOUT_ANALYSIS)
        
        # When - 새로운 매니저 인스턴스로 같은 상태 로드
        new_manager = WorkflowManager(self.config)
        loaded_state = new_manager.get_workflow(workflow_id)
        
        # Then
        assert loaded_state.id == workflow_id
        assert WorkflowStage.LAYOUT_ANALYSIS.value in loaded_state.stages_completed
        assert loaded_state.current_stage == WorkflowStage.TRANSLATION