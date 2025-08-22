# api/auth_api.py
from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

from core.database import DatabaseManager
from core.auth_manager import AuthManager
from core.user_upload_manager import UserUploadManager

logger = logging.getLogger(__name__)

# 전역 인스턴스
db_manager = DatabaseManager()
auth_manager = AuthManager(db_manager)
upload_manager = UserUploadManager(db_manager)

# API 라우터 생성
auth_router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# 요청/응답 모델
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    username: str  # username 또는 email
    password: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[dict] = None

class UserInfo(BaseModel):
    id: int
    username: str
    email: str
    created_at: str
    last_login: Optional[str] = None

# 의존성: 현재 사용자 가져오기
def get_current_user(session_token: str = Cookie(None)):
    """현재 로그인한 사용자 정보 반환"""
    if not session_token:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    
    user = auth_manager.get_user_by_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 세션입니다")
    
    return user

# 선택적 사용자 (로그인하지 않아도 됨)
def get_optional_user(session_token: str = Cookie(None)):
    """로그인한 사용자가 있으면 반환, 없으면 None"""
    if not session_token:
        return None
    return auth_manager.get_user_by_session(session_token)

# 인증 API 엔드포인트들
@auth_router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """사용자 회원가입"""
    success, message = auth_manager.register_user(
        username=request.username,
        email=request.email,
        password=request.password
    )
    
    if success:
        logger.info(f"New user registered: {request.username}")
        return AuthResponse(success=True, message=message)
    else:
        raise HTTPException(status_code=400, detail=message)

@auth_router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, response: Response):
    """사용자 로그인"""
    success, message, session_token = auth_manager.login_user(
        username=request.username,
        password=request.password
    )
    
    if success and session_token:
        # 세션 쿠키 설정
        response.set_cookie(
            key="session_token",
            value=session_token,
            max_age=86400 * 7,  # 7일
            httponly=True,
            secure=False,  # HTTPS에서는 True로 설정
            samesite="lax"
        )
        
        # 사용자 정보 조회
        user = auth_manager.get_user_by_session(session_token)
        user_info = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at,
            "last_login": user.last_login
        } if user else None
        
        logger.info(f"User logged in: {request.username}")
        return AuthResponse(success=True, message=message, user=user_info)
    else:
        raise HTTPException(status_code=401, detail=message)

@auth_router.post("/logout")
async def logout(response: Response, session_token: str = Cookie(None)):
    """사용자 로그아웃"""
    if session_token:
        auth_manager.logout_user(session_token)
    
    # 쿠키 삭제
    response.delete_cookie(key="session_token")
    
    return {"success": True, "message": "로그아웃되었습니다"}

@auth_router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """현재 사용자 정보 조회"""
    return UserInfo(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@auth_router.get("/check")
async def check_auth_status(user = Depends(get_optional_user)):
    """인증 상태 확인"""
    if user:
        return {
            "authenticated": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        }
    else:
        return {"authenticated": False, "user": None}

# api/main.py 수정사항
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Cookie
from fastapi.responses import HTMLResponse, FileResponse
from datetime import datetime
import shutil
from pathlib import Path

# 인증 관련 import
from api.auth_api import auth_router, get_current_user, get_optional_user, upload_manager

# 기존 import들...
from core.workflow_manager import WorkflowManager
from core.error_handler import WorkflowErrorHandler
from core.config import app_config
from core.models import WorkflowStatus, WorkflowStage

# FastAPI 앱에 인증 라우터 추가
app.include_router(auth_router)

# 업로드 엔드포인트 수정 (로그인 필수)
@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)  # 로그인 필수
):
    """문서 업로드 및 번역 시작 (로그인 필요)"""
    
    # 파일 유효성 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{current_user.username}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # 사용자별 출력 디렉토리 생성
    output_dir = OUTPUT_DIR / f"user_{current_user.id}_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 문서 처리 시작
        result = await document_processor.process_document(str(file_path), str(output_dir))
        
        # DB에 업로드 기록 추가
        upload_manager.add_upload_record(
            user_id=current_user.id,
            workflow_id=result["workflow_id"],
            original_filename=file.filename,
            file_size=file.size,
            output_directory=str(output_dir)
        )
        
        return UploadResponse(**result)
        
    except Exception as e:
        # 에러 시 파일 정리
        if file_path.exists():
            file_path.unlink()
        raise e

# 사용자별 업로드 기록 조회 (로그인 필요)
@app.get("/api/v1/my-uploads")
async def get_my_uploads(
    limit: int = 20,
    current_user = Depends(get_current_user)
):
    """내 업로드 기록 조회"""
    
    # 사용자의 업로드 기록 조회
    uploads = upload_manager.get_user_uploads(current_user.id, limit)
    
    # 최신 상태로 업데이트
    for upload in uploads:
        try:
            # 워크플로우 현재 상태 확인
            state = workflow_manager.get_workflow(upload['workflow_id'])
            
            # 상태가 변경된 경우 DB 업데이트
            if (upload['status'] != state.status.value or 
                upload['progress'] != state.progress_percentage):
                
                processing_time = None
                if state.status.value == "COMPLETED":
                    time_diff = state.updated_at - state.created_at
                    processing_time = time_diff.total_seconds()
                
                upload_manager.update_upload_status(
                    workflow_id=upload['workflow_id'],
                    status=state.status.value,
                    progress=state.progress_percentage,
                    processing_time=processing_time,
                    error_message=state.error_info.get('message') if state.error_info else None
                )
                
                # 반환할 데이터 업데이트
                upload['status'] = state.status.value
                upload['progress'] = state.progress_percentage
                upload['processing_time'] = processing_time
                if state.error_info:
                    upload['error_message'] = state.error_info.get('message')
            
        except FileNotFoundError:
            # 워크플로우가 삭제된 경우
            upload['status'] = "DELETED"
            upload['error_message'] = "워크플로우가 삭제되었습니다"
    
    # 통계 정보
    stats = upload_manager.get_user_statistics(current_user.id)
    
    return {
        "uploads": uploads,
        "statistics": stats,
        "total": len(uploads),
        "user": {
            "id": current_user.id,
            "username": current_user.username
        }
    }

# 업로드 기록 삭제 (로그인 필요)
@app.delete("/api/v1/uploads/{workflow_id}")
async def delete_upload_record(
    workflow_id: str,
    current_user = Depends(get_current_user)
):
    """업로드 기록 삭제"""
    
    # 해당 사용자의 업로드인지 확인
    upload = upload_manager.get_upload_by_workflow(workflow_id)
    if not upload or upload['user_id'] != current_user.id:
        raise HTTPException(status_code=404, detail="Upload not found or access denied")
    
    # 워크플로우도 함께 삭제
    try:
        workflow_manager.delete_workflow(workflow_id)
    except FileNotFoundError:
        pass
    
    # DB에서 삭제
    success = upload_manager.delete_upload_record(current_user.id, workflow_id)
    
    if success:
        return {"message": "Upload record deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Upload record not found")

# 업로드 결과 다운로드 (로그인 및 소유권 확인)
@app.get("/api/v1/uploads/{workflow_id}/download")
async def download_upload_result(
    workflow_id: str,
    current_user = Depends(get_current_user)
):
    """업로드 결과 다운로드"""
    
    # 해당 사용자의 업로드인지 확인
    upload = upload_manager.get_upload_by_workflow(workflow_id)
    if not upload or upload['user_id'] != current_user.id:
        raise HTTPException(status_code=404, detail="Upload not found or access denied")
    
    if upload['status'] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Upload not completed yet")
    
    # 기존 다운로드 로직 사용
    return await download_result(workflow_id)

# 로그인 페이지 및 메인 페이지 (수정된 웹 인터페이스)
@app.get("/", response_class=HTMLResponse)
async def web_interface(user = Depends(get_optional_user)):
    """웹 인터페이스 메인 페이지"""
    if user:
        # 로그인된 사용자 - 번역 서비스 페이지
        return HTMLResponse(content=get_main_interface_html())
    else:
        # 비로그인 사용자 - 로그인 페이지
        return HTMLResponse(content=get_login_interface_html())

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """로그인 페이지"""
    return HTMLResponse(content=get_login_interface_html())

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    """회원가입 페이지"""
    return HTMLResponse(content=get_register_interface_html())