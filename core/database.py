# core/database.py
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import json

class DatabaseManager:
    """SQLite 데이터베이스 매니저"""
    
    def __init__(self, db_path: str = "./translation_service.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 및 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- 사용자 테이블
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                );
                
                -- 사용자 세션 테이블
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                -- 업로드 기록 테이블
                CREATE TABLE IF NOT EXISTS upload_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    workflow_id TEXT UNIQUE NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    status TEXT DEFAULT 'CREATED',
                    progress INTEGER DEFAULT 0,
                    output_directory TEXT,
                    processing_time REAL,
                    error_message TEXT,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_time TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                -- 인덱스 생성
                CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_uploads_user ON upload_records(user_id);
                CREATE INDEX IF NOT EXISTS idx_uploads_workflow ON upload_records(workflow_id);
            """)
    
    def get_connection(self):
        """데이터베이스 연결 반환"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        return conn

# core/auth_manager.py
import re
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class User:
    """사용자 데이터 클래스"""
    id: int
    username: str
    email: str
    created_at: str
    last_login: Optional[str] = None
    is_active: bool = True

@dataclass
class UserSession:
    """사용자 세션 데이터 클래스"""
    user_id: int
    session_token: str
    expires_at: str

class AuthManager:
    """사용자 인증 관리"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """비밀번호 해시화"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 반복 횟수
        ).hex()
        
        return password_hash, salt
    
    def _validate_email(self, email: str) -> bool:
        """이메일 형식 검증"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """비밀번호 강도 검증"""
        # 최소 8자, 대소문자, 숫자 포함
        if len(password) < 8:
            return False
        if not re.search(r'[a-z]', password):
            return False
        if not re.search(r'[A-Z]', password):
            return False
        if not re.search(r'\d', password):
            return False
        return True
    
    def register_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """사용자 회원가입"""
        # 입력 검증
        if not username or len(username.strip()) < 3:
            return False, "사용자명은 3자 이상이어야 합니다."
        
        if not self._validate_email(email):
            return False, "올바른 이메일 형식이 아닙니다."
        
        if not self._validate_password(password):
            return False, "비밀번호는 8자 이상, 대소문자와 숫자를 포함해야 합니다."
        
        # 중복 확인
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # 사용자명 중복 확인
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return False, "이미 존재하는 사용자명입니다."
            
            # 이메일 중복 확인
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return False, "이미 등록된 이메일입니다."
            
            # 비밀번호 해시화
            password_hash, salt = self._hash_password(password)
            
            # 사용자 생성
            try:
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, salt)
                    VALUES (?, ?, ?, ?)
                """, (username, email, password_hash, salt))
                
                conn.commit()
                return True, "회원가입이 완료되었습니다."
                
            except sqlite3.Error as e:
                return False, f"회원가입 중 오류가 발생했습니다: {str(e)}"
    
    def login_user(self, username: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """사용자 로그인"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # 사용자 조회 (username 또는 email로 로그인 가능)
            cursor.execute("""
                SELECT id, username, email, password_hash, salt, is_active
                FROM users 
                WHERE (username = ? OR email = ?) AND is_active = 1
            """, (username, username))
            
            user_row = cursor.fetchone()
            if not user_row:
                return False, "존재하지 않는 사용자입니다.", None
            
            # 비밀번호 확인
            password_hash, _ = self._hash_password(password, user_row['salt'])
            if password_hash != user_row['password_hash']:
                return False, "비밀번호가 올바르지 않습니다.", None
            
            # 세션 토큰 생성
            session_token = secrets.token_urlsafe(64)
            expires_at = datetime.now() + timedelta(days=7)  # 7일간 유효
            
            try:
                # 기존 세션 비활성화
                cursor.execute("""
                    UPDATE user_sessions 
                    SET is_active = 0 
                    WHERE user_id = ? AND is_active = 1
                """, (user_row['id'],))
                
                # 새 세션 생성
                cursor.execute("""
                    INSERT INTO user_sessions (user_id, session_token, expires_at)
                    VALUES (?, ?, ?)
                """, (user_row['id'], session_token, expires_at.isoformat()))
                
                # 마지막 로그인 시간 업데이트
                cursor.execute("""
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (user_row['id'],))
                
                conn.commit()
                return True, "로그인 성공", session_token
                
            except sqlite3.Error as e:
                return False, f"로그인 중 오류가 발생했습니다: {str(e)}", None
    
    def get_user_by_session(self, session_token: str) -> Optional[User]:
        """세션 토큰으로 사용자 정보 조회"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT u.id, u.username, u.email, u.created_at, u.last_login, u.is_active
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? 
                AND s.is_active = 1 
                AND s.expires_at > CURRENT_TIMESTAMP
            """, (session_token,))
            
            row = cursor.fetchone()
            if row:
                return User(
                    id=row['id'],
                    username=row['username'],
                    email=row['email'],
                    created_at=row['created_at'],
                    last_login=row['last_login'],
                    is_active=bool(row['is_active'])
                )
            return None
    
    def logout_user(self, session_token: str) -> bool:
        """사용자 로그아웃"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_sessions 
                SET is_active = 0 
                WHERE session_token = ?
            """, (session_token,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def cleanup_expired_sessions(self) -> int:
        """만료된 세션 정리"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_sessions 
                SET is_active = 0 
                WHERE expires_at < CURRENT_TIMESTAMP AND is_active = 1
            """)
            
            conn.commit()
            return cursor.rowcount

# core/user_upload_manager.py
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