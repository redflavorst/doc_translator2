# core/user_upload_manager.py
import sqlite3
from typing import Dict, List, Optional
from .database import DatabaseManager

class UserUploadManager:
    """사용자별 업로드 기록 관리 (DB 기반)"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def add_upload_record(self, user_id: int, workflow_id: str, 
                         original_filename: str, file_size: int, 
                         output_directory: str) -> bool:
        """새 업로드 기록 추가"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO upload_records 
                    (user_id, workflow_id, original_filename, file_size, output_directory)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, workflow_id, original_filename, file_size, output_directory))
                
                conn.commit()
                return True
                
            except sqlite3.Error as e:
                print(f"Failed to add upload record: {e}")
                return False
    
    def update_upload_status(self, workflow_id: str, status: str, 
                           progress: int = 0, processing_time: float = None,
                           error_message: str = None) -> bool:
        """업로드 상태 업데이트"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            update_fields = ["status = ?", "progress = ?"]
            values = [status, progress]
            
            if processing_time is not None:
                update_fields.append("processing_time = ?")
                values.append(processing_time)
            
            if error_message is not None:
                update_fields.append("error_message = ?")
                values.append(error_message)
            
            if status == "COMPLETED":
                update_fields.append("completed_time = CURRENT_TIMESTAMP")
            
            values.append(workflow_id)
            
            try:
                cursor.execute(f"""
                    UPDATE upload_records 
                    SET {', '.join(update_fields)}
                    WHERE workflow_id = ?
                """, values)
                
                conn.commit()
                return cursor.rowcount > 0
                
            except sqlite3.Error as e:
                print(f"Failed to update upload status: {e}")
                return False
    
    def get_user_uploads(self, user_id: int, limit: int = 20) -> List[Dict]:
        """사용자의 업로드 기록 조회"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, workflow_id, original_filename, file_size, status, 
                       progress, output_directory, processing_time, error_message,
                       upload_time, completed_time
                FROM upload_records 
                WHERE user_id = ? 
                ORDER BY upload_time DESC 
                LIMIT ?
            """, (user_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_upload_by_workflow(self, workflow_id: str) -> Optional[Dict]:
        """워크플로우 ID로 업로드 기록 조회"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM upload_records WHERE workflow_id = ?
            """, (workflow_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def delete_upload_record(self, user_id: int, workflow_id: str) -> bool:
        """업로드 기록 삭제"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM upload_records 
                WHERE user_id = ? AND workflow_id = ?
            """, (user_id, workflow_id))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_user_statistics(self, user_id: int) -> Dict:
        """사용자별 통계"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_uploads,
                    SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status IN ('RUNNING', 'CREATED') THEN 1 ELSE 0 END) as processing,
                    SUM(file_size) as total_files_size,
                    AVG(CASE WHEN processing_time IS NOT NULL THEN processing_time END) as avg_processing_time
                FROM upload_records 
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "total_uploads": row['total_uploads'] or 0,
                    "completed": row['completed'] or 0,
                    "failed": row['failed'] or 0,
                    "processing": row['processing'] or 0,
                    "total_files_size": row['total_files_size'] or 0,
                    "avg_processing_time": round(row['avg_processing_time'] or 0, 2)
                }
            
            return {
                "total_uploads": 0,
                "completed": 0,
                "failed": 0,
                "processing": 0,
                "total_files_size": 0,
                "avg_processing_time": 0
            }