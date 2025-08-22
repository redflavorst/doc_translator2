# core/user_history_manager.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib

@dataclass
class UploadRecord:
    """업로드 기록 데이터 클래스"""
    id: str                    # 업로드 고유 ID
    workflow_id: str           # 워크플로우 ID
    original_filename: str     # 원본 파일명
    file_size: int            # 파일 크기 (bytes)
    upload_time: datetime     # 업로드 시간
    status: str               # 현재 상태
    progress: int             # 진행률 (0-100)
    output_directory: str     # 출력 디렉토리
    processing_time: Optional[float] = None  # 처리 시간 (초)
    error_message: Optional[str] = None      # 에러 메시지
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        data = asdict(self)
        data['upload_time'] = self.upload_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UploadRecord':
        """딕셔너리에서 복원"""
        data['upload_time'] = datetime.fromisoformat(data['upload_time'])
        return cls(**data)

class UserHistoryManager:
    """사용자별 업로드 기록 관리"""
    
    def __init__(self, history_dir: str = "./user_histories"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        
    def _get_user_history_file(self, session_id: str) -> Path:
        """사용자별 히스토리 파일 경로"""
        # 세션 ID를 해시화하여 파일명으로 사용
        session_hash = hashlib.md5(session_id.encode()).hexdigest()[:16]
        return self.history_dir / f"user_{session_hash}.json"
    
    def _load_user_history(self, session_id: str) -> List[UploadRecord]:
        """사용자 히스토리 로드"""
        history_file = self._get_user_history_file(session_id)
        
        if not history_file.exists():
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [UploadRecord.from_dict(item) for item in data]
        except Exception as e:
            print(f"Failed to load user history: {e}")
            return []
    
    def _save_user_history(self, session_id: str, records: List[UploadRecord]):
        """사용자 히스토리 저장"""
        history_file = self._get_user_history_file(session_id)
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                data = [record.to_dict() for record in records]
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save user history: {e}")
    
    def add_upload_record(self, session_id: str, workflow_id: str, 
                         original_filename: str, file_size: int, 
                         output_directory: str) -> str:
        """새 업로드 기록 추가"""
        upload_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{workflow_id[:8]}"
        
        record = UploadRecord(
            id=upload_id,
            workflow_id=workflow_id,
            original_filename=original_filename,
            file_size=file_size,
            upload_time=datetime.now(),
            status="CREATED",
            progress=0,
            output_directory=output_directory
        )
        
        # 기존 히스토리 로드
        history = self._load_user_history(session_id)
        
        # 새 기록 추가 (최신순)
        history.insert(0, record)
        
        # 최대 50개 기록만 유지
        if len(history) > 50:
            history = history[:50]
        
        # 저장
        self._save_user_history(session_id, history)
        
        return upload_id
    
    def update_upload_status(self, session_id: str, workflow_id: str, 
                           status: str, progress: int = 0, 
                           processing_time: float = None,
                           error_message: str = None):
        """업로드 상태 업데이트"""
        history = self._load_user_history(session_id)
        
        for record in history:
            if record.workflow_id == workflow_id:
                record.status = status
                record.progress = progress
                if processing_time is not None:
                    record.processing_time = processing_time
                if error_message is not None:
                    record.error_message = error_message
                break
        
        self._save_user_history(session_id, history)
    
    def get_user_history(self, session_id: str, limit: int = 20) -> List[UploadRecord]:
        """사용자 업로드 히스토리 조회"""
        history = self._load_user_history(session_id)
        return history[:limit]
    
    def get_upload_record(self, session_id: str, upload_id: str) -> Optional[UploadRecord]:
        """특정 업로드 기록 조회"""
        history = self._load_user_history(session_id)
        
        for record in history:
            if record.id == upload_id:
                return record
        
        return None
    
    def delete_upload_record(self, session_id: str, upload_id: str) -> bool:
        """업로드 기록 삭제"""
        history = self._load_user_history(session_id)
        
        for i, record in enumerate(history):
            if record.id == upload_id:
                del history[i]
                self._save_user_history(session_id, history)
                return True
        
        return False
    
    def get_statistics(self, session_id: str) -> Dict:
        """사용자별 통계"""
        history = self._load_user_history(session_id)
        
        if not history:
            return {
                "total_uploads": 0,
                "completed": 0,
                "failed": 0,
                "processing": 0,
                "total_files_size": 0,
                "avg_processing_time": 0
            }
        
        completed = [r for r in history if r.status == "COMPLETED"]
        failed = [r for r in history if r.status == "FAILED"]
        processing = [r for r in history if r.status in ["RUNNING", "CREATED"]]
        
        total_size = sum(r.file_size for r in history)
        processing_times = [r.processing_time for r in completed if r.processing_time]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_uploads": len(history),
            "completed": len(completed),
            "failed": len(failed),
            "processing": len(processing),
            "total_files_size": total_size,
            "avg_processing_time": round(avg_time, 2)
        }
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """오래된 기록 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for history_file in self.history_dir.glob("user_*.json"):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                original_count = len(data)
                # 최근 기록만 유지
                recent_data = [
                    item for item in data 
                    if datetime.fromisoformat(item['upload_time']) > cutoff_date
                ]
                
                if len(recent_data) < original_count:
                    with open(history_file, 'w', encoding='utf-8') as f:
                        json.dump(recent_data, f, indent=2, ensure_ascii=False)
                    
                    cleaned_count += (original_count - len(recent_data))
                    
            except Exception as e:
                print(f"Error cleaning {history_file}: {e}")
        
        return cleaned_count