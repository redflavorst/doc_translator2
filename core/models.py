# core/models.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class WorkflowStatus(Enum):
    """워크플로우 실행 상태"""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class WorkflowStage(Enum):
    """워크플로우 실행 단계"""
    LAYOUT_ANALYSIS = "LAYOUT_ANALYSIS"
    TRANSLATION = "TRANSLATION"
    PDF_EXPORT = "PDF_EXPORT"
    COMPLETION = "COMPLETION"


@dataclass
class WorkflowState:
    """워크플로우 상태를 관리하는 모델"""
    
    id: str
    status: WorkflowStatus
    current_stage: WorkflowStage
    stages_completed: List[str]
    input_file_path: str
    output_directory: str
    error_info: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    progress_percentage: int = 0
    
    # 상세 진행 정보
    current_page: int = 0
    total_pages: int = 0
    current_action: str = ""  # 현재 수행 중인 작업 설명
    stage_details: Optional[Dict[str, Any]] = None  # 단계별 상세 정보
    
    # 페이지별 진행 추적
    pages_analyzed: int = 0  # 레이아웃 분석 완료된 페이지 수
    pages_translated: int = 0  # 번역 완료된 페이지 수
    
    def __post_init__(self):
        """데이터클래스 초기화 후 처리"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def complete_stage(self, stage: WorkflowStage) -> None:
        """특정 단계를 완료 처리"""
        stage_value = stage.value
        
        # 중복 추가 방지
        if stage_value not in self.stages_completed:
            self.stages_completed.append(stage_value)
        
        # 단계 완료 시 페이지 수 업데이트
        if stage == WorkflowStage.LAYOUT_ANALYSIS:
            self.pages_analyzed = self.total_pages
        elif stage == WorkflowStage.TRANSLATION:
            self.pages_translated = self.total_pages
        
        # 다음 단계로 이동
        self._move_to_next_stage(stage)
        
        # 진행률 업데이트
        self._update_progress()
        
        # 업데이트 시간 갱신
        self.updated_at = datetime.now()
    
    def _move_to_next_stage(self, completed_stage: WorkflowStage) -> None:
        """완료된 단계에 따라 다음 단계로 이동"""
        stage_flow = {
            WorkflowStage.LAYOUT_ANALYSIS: WorkflowStage.TRANSLATION,
            WorkflowStage.TRANSLATION: WorkflowStage.COMPLETION,
            WorkflowStage.COMPLETION: WorkflowStage.COMPLETION  # 마지막 단계
        }
        
        next_stage = stage_flow.get(completed_stage)
        if next_stage:
            self.current_stage = next_stage
            
        # 모든 단계 완료 시 상태 변경
        if completed_stage == WorkflowStage.COMPLETION:
            self.status = WorkflowStatus.COMPLETED
            self.progress_percentage = 100
    
    def _update_progress(self) -> None:
        """페이지 기반 진행률 계산"""
        if self.total_pages == 0:
            # 페이지 수를 모르는 경우 단계 기반 계산
            total_stages = len(WorkflowStage)
            completed_count = len(self.stages_completed)
            self.progress_percentage = min(int((completed_count / total_stages) * 100), 100)
        else:
            # 더 정확한 계산: 각 페이지마다 레이아웃 분석 + 번역 = 2개 작업
            # 총 작업 수 = 페이지 수 × 2
            total_tasks = self.total_pages * 2
            
            # 완료된 작업 수 = 분석된 페이지 + 번역된 페이지
            completed_tasks = self.pages_analyzed + self.pages_translated
            
            # 진행률 = (완료된 작업 / 총 작업) × 100
            self.progress_percentage = min(int((completed_tasks / total_tasks) * 100), 100)
    
    def set_error(self, error_type: str, message: str, traceback: str = None) -> None:
        """에러 정보 설정"""
        self.status = WorkflowStatus.FAILED
        self.error_info = {
            "error_type": error_type,
            "message": message,
            "traceback": traceback,
            "occurred_at": datetime.now().isoformat()
        }
        self.updated_at = datetime.now()
    
    def pause(self) -> None:
        """워크플로우 일시 정지"""
        if self.status == WorkflowStatus.RUNNING:
            self.status = WorkflowStatus.PAUSED
            self.updated_at = datetime.now()
    
    def resume(self) -> None:
        """워크플로우 재시작"""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING
            self.updated_at = datetime.now()
    
    def start(self) -> None:
        """워크플로우 시작"""
        if self.status == WorkflowStatus.CREATED:
            self.status = WorkflowStatus.RUNNING
            self.updated_at = datetime.now()
    
    def validate(self) -> None:
        """워크플로우 상태 유효성 검증"""
        if not self.id or self.id.strip() == "":
            raise ValueError("Workflow ID cannot be empty")
        
        if not self.input_file_path or self.input_file_path.strip() == "":
            raise ValueError("Input file path cannot be empty")
        
        if not self.output_directory or self.output_directory.strip() == "":
            raise ValueError("Output directory cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 직렬화"""
        data = asdict(self)
        
        # Enum을 문자열로 변환
        data['status'] = self.status.value
        data['current_stage'] = self.current_stage.value
        
        # datetime을 ISO 형식 문자열로 변환
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        
        # 페이지 진행 정보 포함
        data['pages_analyzed'] = getattr(self, 'pages_analyzed', 0)
        data['pages_translated'] = getattr(self, 'pages_translated', 0)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """딕셔너리에서 복원"""
        # Enum 변환
        data['status'] = WorkflowStatus(data['status'])
        data['current_stage'] = WorkflowStage(data['current_stage'])
        
        # datetime 변환
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """JSON 문자열로 직렬화"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WorkflowState':
        """JSON 문자열에서 복원"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def is_recoverable(self) -> bool:
        """복구 가능한 상태인지 확인"""
        return self.status in [WorkflowStatus.FAILED, WorkflowStatus.PAUSED]
    
    def can_retry_stage(self, stage: WorkflowStage) -> bool:
        """특정 단계를 재시도할 수 있는지 확인"""
        return (
            self.status == WorkflowStatus.FAILED and 
            self.current_stage == stage and 
            stage.value not in self.stages_completed
        )
    
    def get_next_stage(self) -> Optional[WorkflowStage]:
        """다음 실행할 단계 반환"""
        if self.status == WorkflowStatus.COMPLETED:
            return None
        
        # 현재 단계가 완료되지 않았다면 현재 단계 반환
        if self.current_stage.value not in self.stages_completed:
            return self.current_stage
        
        # 다음 단계 계산
        stage_order = [
            WorkflowStage.LAYOUT_ANALYSIS,
            WorkflowStage.TRANSLATION,
            WorkflowStage.COMPLETION
        ]
        
        try:
            current_index = stage_order.index(self.current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            pass
        
        return None