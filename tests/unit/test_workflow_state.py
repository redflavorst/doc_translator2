# tests/unit/test_workflow_state.py
import sys
from pathlib import Path
# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from datetime import datetime
from core.models import WorkflowState, WorkflowStatus, WorkflowStage


class TestWorkflowState:
    """워크플로우 상태 모델 테스트"""
    
    def test_workflow_state_creation(self):
        """워크플로우 상태 생성 테스트"""
        # Given
        workflow_id = "test-123"
        input_file = "/path/to/test.pdf"
        output_dir = "/path/to/output"
        
        # When
        state = WorkflowState(
            id=workflow_id,
            status=WorkflowStatus.CREATED,
            current_stage=WorkflowStage.LAYOUT_ANALYSIS,
            stages_completed=[],
            input_file_path=input_file,
            output_directory=output_dir
        )
        
        # Then
        assert state.id == workflow_id
        assert state.status == WorkflowStatus.CREATED
        assert state.current_stage == WorkflowStage.LAYOUT_ANALYSIS
        assert state.stages_completed == []
        assert state.input_file_path == input_file
        assert state.output_directory == output_dir
        assert state.error_info is None
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)
        assert state.progress_percentage == 0

    def test_workflow_state_with_error(self):
        """에러 정보가 있는 워크플로우 상태 테스트"""
        # Given
        error_info = {
            "error_type": "OCRError",
            "message": "Failed to process PDF",
            "traceback": "..."
        }
        
        # When
        state = WorkflowState(
            id="test-456",
            status=WorkflowStatus.FAILED,
            current_stage=WorkflowStage.LAYOUT_ANALYSIS,
            stages_completed=[],
            input_file_path="/test.pdf",
            output_directory="/output",
            error_info=error_info
        )
        
        # Then
        assert state.status == WorkflowStatus.FAILED
        assert state.error_info == error_info

    def test_workflow_state_progress_update(self):
        """워크플로우 진행률 업데이트 테스트"""
        # Given
        state = WorkflowState(
            id="test-789",
            status=WorkflowStatus.RUNNING,
            current_stage=WorkflowStage.TRANSLATION,
            stages_completed=["LAYOUT_ANALYSIS"],
            input_file_path="/test.pdf",
            output_directory="/output",
            progress_percentage=50
        )
        
        # Then
        assert state.progress_percentage == 50
        assert "LAYOUT_ANALYSIS" in state.stages_completed
        assert state.current_stage == WorkflowStage.TRANSLATION

    def test_workflow_state_serialization(self):
        """워크플로우 상태 직렬화 테스트"""
        # Given
        state = WorkflowState(
            id="test-serialize",
            status=WorkflowStatus.RUNNING,
            current_stage=WorkflowStage.LAYOUT_ANALYSIS,
            stages_completed=[],
            input_file_path="/test.pdf",
            output_directory="/output"
        )
        
        # When
        state_dict = state.to_dict()
        restored_state = WorkflowState.from_dict(state_dict)
        
        # Then
        assert restored_state.id == state.id
        assert restored_state.status == state.status
        assert restored_state.current_stage == state.current_stage
        assert restored_state.input_file_path == state.input_file_path

    def test_workflow_state_stage_completion(self):
        """단계 완료 처리 테스트"""
        # Given
        state = WorkflowState(
            id="test-complete",
            status=WorkflowStatus.RUNNING,
            current_stage=WorkflowStage.LAYOUT_ANALYSIS,
            stages_completed=[],
            input_file_path="/test.pdf",
            output_directory="/output"
        )
        
        # When
        state.complete_stage(WorkflowStage.LAYOUT_ANALYSIS)
        
        # Then
        assert WorkflowStage.LAYOUT_ANALYSIS.value in state.stages_completed
        assert state.current_stage == WorkflowStage.TRANSLATION
        assert state.progress_percentage == 33  # 3단계 중 1단계 완료 (33%)

    def test_workflow_state_validation(self):
        """워크플로우 상태 유효성 검증 테스트"""
        # Given
        state = WorkflowState(
            id="",  # 빈 ID
            status=WorkflowStatus.CREATED,
            current_stage=WorkflowStage.LAYOUT_ANALYSIS,
            stages_completed=[],
            input_file_path="",  # 빈 파일 경로
            output_directory="/output"
        )
        
        # When & Then
        with pytest.raises(ValueError, match="Workflow ID cannot be empty"):
            state.validate()

    def test_workflow_status_enum_values(self):
        """워크플로우 상태 Enum 값 테스트"""
        assert WorkflowStatus.CREATED.value == "CREATED"
        assert WorkflowStatus.RUNNING.value == "RUNNING"
        assert WorkflowStatus.COMPLETED.value == "COMPLETED"
        assert WorkflowStatus.FAILED.value == "FAILED"
        assert WorkflowStatus.PAUSED.value == "PAUSED"

    def test_workflow_stage_enum_values(self):
        """워크플로우 단계 Enum 값 테스트"""
        assert WorkflowStage.LAYOUT_ANALYSIS.value == "LAYOUT_ANALYSIS"
        assert WorkflowStage.TRANSLATION.value == "TRANSLATION"
        assert WorkflowStage.COMPLETION.value == "COMPLETION"