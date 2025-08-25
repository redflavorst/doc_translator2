# api/main.py
import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import tempfile
import asyncio
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Cookie, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import uuid

from core.workflow_manager import WorkflowManager
from core.error_handler import WorkflowErrorHandler
from core.config import app_config
from core.models import WorkflowStatus, WorkflowStage

# 새로 추가된 인증 관련 import
from core.database import DatabaseManager
from core.auth_manager import AuthManager
from core.user_upload_manager import UserUploadManager

# 기존 서비스 파일들 import
from services.layout_analysis_service import LayoutAnalysisService, LayoutAnalysisResult
# 페이지별 레이아웃 분석 서비스 (테스트용)
from services.layout_analysis_service_paged import LayoutAnalysisServicePaged
from services.translation_service import TranslationService, BatchTranslationResult
from core.user_history_manager import UserHistoryManager

# 로거 설정
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Document Translation Service",
    description="PDF 문서를 한국어로 번역하는 서비스",
    version="1.0.0"
)

# 전역 서비스 인스턴스
workflow_manager = WorkflowManager(app_config.workflow)
error_handler = WorkflowErrorHandler(workflow_manager, app_config.workflow)
history_manager = UserHistoryManager()

# 새로 추가: 인증 관련 전역 인스턴스
db_manager = DatabaseManager()
auth_manager = AuthManager(db_manager)
upload_manager = UserUploadManager(db_manager)

# 기존 서비스 인스턴스 생성 (페이지별 처리 방식으로 변경)
# layout_service = LayoutAnalysisService({  # 기존 방식
#     'use_gpu': app_config.layout_analysis.use_gpu,
#     'det_limit_side_len': 1920,
#     'use_table': True
# })

# 페이지별 처리 방식 사용 (실시간 진행률 추적 가능)
layout_service = LayoutAnalysisServicePaged({
    'use_gpu': app_config.layout_analysis.use_gpu,
    'use_table': True
})

translation_service = TranslationService({
    'model_name': app_config.translation.model_name,
    'temperature': app_config.translation.temperature,
    'ollama_url': app_config.translation.ollama_base_url,
    'quality_threshold': app_config.translation.quality_threshold,
    'max_retries': 3
})

# 업로드 디렉토리 설정
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# 응답 모델들
class UploadResponse(BaseModel):
    workflow_id: str
    status: str
    message: str


class WorkflowStatusResponse(BaseModel):
    id: str
    status: str
    current_stage: str
    progress_percentage: int
    created_at: str
    updated_at: str
    error_info: Optional[dict] = None
    
    # 상세 진행 정보
    current_page: int = 0
    total_pages: int = 0
    current_action: str = ""
    stage_details: Optional[dict] = None


class WorkflowListResponse(BaseModel):
    total: int
    running: int
    completed: int
    failed: int
    paused: int

# 인증 관련 의존성 함수들
def get_current_user(session_token: str = Cookie(None)):
    """현재 로그인한 사용자 정보 반환"""
    if not session_token:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    
    user = auth_manager.get_user_by_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 세션입니다")
    
    return user

def get_optional_user(session_token: str = Cookie(None)):
    """로그인한 사용자가 있으면 반환, 없으면 None"""
    if not session_token:
        return None
    return auth_manager.get_user_by_session(session_token)

# 문서 처리 클래스 (기존과 동일하지만 사용자 정보 추가)
class DocumentProcessor:
    """문서 처리 오케스트레이터"""
    
    def __init__(self, workflow_manager: WorkflowManager, 
                 error_handler: WorkflowErrorHandler,
                 layout_service: LayoutAnalysisService,
                 translation_service: TranslationService):
        self.workflow_manager = workflow_manager
        self.error_handler = error_handler
        self.layout_service = layout_service
        self.translation_service = translation_service
    
    async def process_document(self, file_path: str, output_dir: str) -> dict:
        """
        문서 처리 메인 로직
        
        Args:
            file_path: PDF 파일 경로
            output_dir: 출력 디렉토리
            
        Returns:
            처리 결과
        """
        # 워크플로우 생성
        workflow_id = self.workflow_manager.create_workflow(file_path, output_dir)
        
        try:
            # 백그라운드에서 실제 처리 시작
            asyncio.create_task(self._process_workflow(workflow_id))
            
            return {
                "workflow_id": workflow_id,
                "status": "processing",
                "message": "Document processing started"
            }
            
        except Exception as e:
            self.workflow_manager.set_workflow_error(
                workflow_id, type(e).__name__, str(e)
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_workflow(self, workflow_id: str):
        """
        워크플로우 백그라운드 처리
        
        Args:
            workflow_id: 워크플로우 ID
        """
        try:
            # 워크플로우 시작
            self.workflow_manager.update_workflow_status(workflow_id, WorkflowStatus.RUNNING)
            
            state = self.workflow_manager.get_workflow(workflow_id)
            
            # 1단계: 레이아웃 분석
            await self._execute_layout_analysis(workflow_id, state.input_file_path, state.output_directory)
            
            # 2단계: 번역
            await self._execute_translation(workflow_id, state.output_directory)
            
            # 완료 처리
            self.workflow_manager.complete_stage(workflow_id, WorkflowStage.COMPLETION)
            
        except Exception as e:
            self.workflow_manager.set_workflow_error(
                workflow_id, type(e).__name__, str(e)
            )
    
    async def _execute_layout_analysis(self, workflow_id: str, input_file: str, output_dir: str):
        """레이아웃 분석 실행"""
        # 진행 상황 업데이트: 레이아웃 분석 시작
        self.workflow_manager.update_progress(
            workflow_id, 
            current_action="PDF 문서 페이지 수 확인 중...",
            stage_details={"stage": "layout_analysis", "status": "starting"}
        )
        
        # 진행 상황 콜백 함수 정의
        def progress_callback(current_page, total_pages, action):
            # 모델 로드 중인지 확인
            if "모델 로드" in action:
                self.workflow_manager.update_progress(
                    workflow_id,
                    current_action=action,
                    stage_details={
                        "stage": "layout_analysis",
                        "status": "loading_model"
                    }
                )
            else:
                # 레이아웃 분석 중 페이지별 진행 상황 업데이트
                self.workflow_manager.update_progress(
                    workflow_id,
                    current_page=current_page,
                    total_pages=total_pages,
                    pages_analyzed=current_page,
                    current_action=f"레이아웃 분석: {action}",
                    stage_details={
                        "stage": "layout_analysis",
                        "status": "processing",
                        "current_page": current_page,
                        "total_pages": total_pages
                    }
                )
        
        # 비동기적으로 레이아웃 분석 실행 (백그라운드 스레드에서)
        import asyncio
        import concurrent.futures
        
        try:
            # 스레드 풀에서 동기 함수 실행
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    self.layout_service.analyze_document,
                    input_file,
                    output_dir,
                    progress_callback
                )
            
            if not result.success:
                raise RuntimeError(f"Layout analysis failed: {result.error}")
                
        except Exception as e:
            # 에러 처리
            self.workflow_manager.set_workflow_error(
                workflow_id, 
                type(e).__name__, 
                str(e)
            )
            raise
        
        # 페이지 수 업데이트 및 레이아웃 분석 완료
        if result.pages:
            total_pages = len(result.pages)
            # 전체 페이지 수 설정 및 레이아웃 분석 완료 표시
            self.workflow_manager.update_progress(
                workflow_id,
                total_pages=total_pages,
                pages_analyzed=total_pages,  # 모든 페이지 분석 완료
                current_action="레이아웃 분석 완료",
                stage_details={
                    "stage": "layout_analysis", 
                    "status": "completed",
                    "pages_found": total_pages,
                    "markdown_files": len(result.markdown_files)
                }
            )
        
        # 단계 완료 처리
        self.workflow_manager.complete_stage(workflow_id, WorkflowStage.LAYOUT_ANALYSIS)
        return result
    
    async def _execute_translation(self, workflow_id: str, output_dir: str):
        """번역 실행"""
        # 마크다운 파일들 찾기
        markdown_files = list(Path(output_dir).glob("page_*.md"))
        # 이미 번역된 파일 제외
        markdown_files = [f for f in markdown_files if not str(f).endswith('_korean.md')]
        
        if not markdown_files:
            raise FileNotFoundError("No markdown files found for translation")
        
        total_files = len(markdown_files)
        self.workflow_manager.update_progress(
            workflow_id,
            current_action=f"총 {total_files}개 파일 번역 준비 중...",
            stage_details={"stage": "translation", "status": "preparing", "total_files": total_files}
        )
        
        # 각 파일을 개별적으로 번역하면서 진행 상황 업데이트
        translated_files = []
        failed_files = []
        
        for idx, md_file in enumerate(markdown_files, 1):
            # 진행 상황 업데이트 (페이지별 번역 진행)
            self.workflow_manager.update_progress(
                workflow_id,
                current_page=idx,
                pages_translated=idx - 1,  # 현재까지 완료된 페이지 수
                current_action=f"번역 중: {md_file.name} ({idx}/{total_files})",
                stage_details={
                    "stage": "translation",
                    "status": "processing",
                    "current_file": str(md_file.name),
                    "progress": f"{idx}/{total_files}",
                    "percentage": int((idx / total_files) * 100)
                }
            )
            
            # 개별 파일 번역
            try:
                result = self.translation_service.translate_document(str(md_file))
                if result.success:
                    translated_files.append(result.output_file)
                    # 번역 성공 시 완료된 페이지 수 업데이트
                    self.workflow_manager.update_progress(
                        workflow_id,
                        pages_translated=idx  # 현재 페이지까지 번역 완료
                    )
                else:
                    failed_files.append(str(md_file))
            except Exception as e:
                logger.warning(f"Translation failed for {md_file}: {e}")
                failed_files.append(str(md_file))
            
            # 비동기 처리를 위한 짧은 대기
            await asyncio.sleep(0.1)
        
        # 완료 상태 업데이트
        self.workflow_manager.update_progress(
            workflow_id,
            pages_translated=len(translated_files),  # 최종 번역 완료 페이지 수
            current_action="번역 완료",
            stage_details={
                "stage": "translation",
                "status": "completed",
                "translated_files": len(translated_files),
                "failed_files": len(failed_files)
            }
        )
        
        # 단계 완료 처리
        self.workflow_manager.complete_stage(workflow_id, WorkflowStage.TRANSLATION)
        
        return {
            "translated_files": translated_files,
            "failed_files": failed_files,
            "success": len(failed_files) == 0
        }


# 문서 처리기 인스턴스
document_processor = DocumentProcessor(
    workflow_manager, error_handler, layout_service, translation_service
)

# ========== 인증 API 엔드포인트들 ==========

# 회원가입
@app.post("/api/v1/auth/register")
async def register_user(request: dict):
    """사용자 회원가입"""
    username = request.get('username')
    email = request.get('email')
    password = request.get('password')
    
    if not all([username, email, password]):
        raise HTTPException(status_code=400, detail="모든 필드를 입력해주세요")
    
    success, message = auth_manager.register_user(username, email, password)
    
    if success:
        logger.info(f"New user registered: {username}")
        return {"success": True, "message": message}
    else:
        raise HTTPException(status_code=400, detail=message)

# 로그인
@app.post("/api/v1/auth/login")
async def login_user(request: dict, response: Response):
    """사용자 로그인"""
    username = request.get('username')
    password = request.get('password')
    
    if not all([username, password]):
        raise HTTPException(status_code=400, detail="사용자명과 비밀번호를 입력해주세요")
    
    success, message, session_token = auth_manager.login_user(username, password)
    
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
        
        logger.info(f"User logged in: {username}")
        return {"success": True, "message": message, "user": user_info}
    else:
        raise HTTPException(status_code=401, detail=message)

# 로그아웃
@app.post("/api/v1/auth/logout")
async def logout_user(response: Response, session_token: str = Cookie(None)):
    """사용자 로그아웃"""
    if session_token:
        auth_manager.logout_user(session_token)
    
    # 쿠키 삭제
    response.delete_cookie(key="session_token")
    
    return {"success": True, "message": "로그아웃되었습니다"}

# 인증 상태 확인
@app.get("/api/v1/auth/check")
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

# 내 정보 조회
@app.get("/api/v1/auth/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    """현재 사용자 정보 조회"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

 # API 엔드포인트들
 # ========== 업로드 API 엔드포인트들 (인증 필요) ==========
@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
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
        try:
            file_size = file_path.stat().st_size if file_path.exists() else 0
            upload_manager.add_upload_record(
                user_id=current_user.id,
                workflow_id=result["workflow_id"],
                original_filename=file.filename,
                file_size=file_size,
                output_directory=str(output_dir)
            )
        except Exception as e:
            logger.warning(f"Failed to add upload record: {e}")
        
        return UploadResponse(**result)
        
    except Exception as e:
        # 에러 시 파일 정리
        if file_path.exists():
            file_path.unlink()
        raise e


@app.get("/api/v1/workflows/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    워크플로우 상태 조회
    """
    try:
        state = workflow_manager.get_workflow(workflow_id)
        return WorkflowStatusResponse(
            id=state.id,
            status=state.status.value,
            current_stage=state.current_stage.value,
            progress_percentage=state.progress_percentage,
            created_at=state.created_at.isoformat(),
            updated_at=state.updated_at.isoformat(),
            error_info=state.error_info,
            current_page=state.current_page,
            total_pages=state.total_pages,
            current_action=state.current_action,
            stage_details=state.stage_details
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@app.get("/api/v1/workflows/", response_model=WorkflowListResponse)
async def list_workflows():
    """
    워크플로우 목록 및 통계 조회
    """
    stats = workflow_manager.get_workflow_statistics()
    return WorkflowListResponse(**stats)


@app.get("/api/v1/my-uploads")
async def get_my_uploads(
    limit: int = 20,
    current_user = Depends(get_current_user)
):
    """내 업로드 기록 조회 (DB 기반)"""
    
    uploads = upload_manager.get_user_uploads(current_user.id, limit)
    
    for upload in uploads:
        try:
            state = workflow_manager.get_workflow(upload['workflow_id'])
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
                
                upload['status'] = state.status.value
                upload['progress'] = state.progress_percentage
                upload['processing_time'] = processing_time
                if state.error_info:
                    upload['error_message'] = state.error_info.get('message')
        except FileNotFoundError:
            upload['status'] = "DELETED"
            upload['error_message'] = "워크플로우가 삭제되었습니다"
    
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


@app.delete("/api/v1/uploads/{workflow_id}")
async def delete_upload_record(
    workflow_id: str,
    current_user = Depends(get_current_user)
):
    """업로드 기록 삭제 (로그인 필요)"""
    upload = upload_manager.get_upload_by_workflow(workflow_id)
    if not upload or upload['user_id'] != current_user.id:
        raise HTTPException(status_code=404, detail="Upload not found or access denied")
    
    try:
        workflow_manager.delete_workflow(workflow_id)
    except FileNotFoundError:
        pass
    
    success = upload_manager.delete_upload_record(current_user.id, workflow_id)
    if success:
        return {"message": "Upload record deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Upload record not found")


@app.get("/api/v1/uploads/{workflow_id}/download")
async def download_upload_result(
    workflow_id: str,
    current_user = Depends(get_current_user)
):
    """업로드 결과 다운로드 (로그인 및 소유권 확인)"""
    upload = upload_manager.get_upload_by_workflow(workflow_id)
    if not upload or upload['user_id'] != current_user.id:
        raise HTTPException(status_code=404, detail="Upload not found or access denied")
    
    if upload['status'] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Upload not completed yet")
    
    return await download_result(workflow_id)


@app.get("/api/v1/workflows/{workflow_id}/download")
async def download_result(workflow_id: str):
    """
    번역 결과 다운로드 (ZIP 파일)
    """
    try:
        state = workflow_manager.get_workflow(workflow_id)
        
        if state.status != WorkflowStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail="Workflow is not completed yet"
            )
        
        output_dir = Path(state.output_directory)
        if not output_dir.exists():
            raise HTTPException(status_code=404, detail="Output files not found")
        
        # ZIP 파일 생성
        zip_path = output_dir.parent / f"{workflow_id}_result.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)
        
        return FileResponse(
            path=zip_path,
            filename=f"translation_result_{workflow_id}.zip",
            media_type="application/zip"
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@app.delete("/api/v1/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    워크플로우 삭제
    """
    try:
        workflow_manager.delete_workflow(workflow_id)
        return {"message": "Workflow deleted successfully"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@app.get("/health")
async def health_check():
    """
    헬스 체크
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# 7. 웹 인터페이스 수정 (기존 @app.get("/") 함수 수정)
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

def get_register_interface_html() -> str:
    """회원가입 페이지 HTML (필요시 별도 페이지용)"""
    return get_login_interface_html()  # 로그인 페이지에 탭으로 포함되어 있음

def get_login_interface_html() -> str:

    """로그인/회원가입 페이지 HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔐 문서 번역 서비스 - 로그인</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .auth-container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.2);
                width: 100%;
                max-width: 450px;
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }
            .header h1 { font-size: 28px; margin-bottom: 10px; }
            .header p { opacity: 0.9; font-size: 16px; }
            .auth-tabs {
                display: flex;
                border-bottom: 2px solid #f0f0f0;
            }
            .tab {
                flex: 1;
                padding: 20px;
                background: #f8f9fa;
                cursor: pointer;
                text-align: center;
                font-weight: 600;
                color: #666;
                transition: all 0.3s;
                border: none;
                font-size: 16px;
            }
            .tab.active {
                background: white;
                color: #667eea;
                border-bottom: 3px solid #667eea;
            }
            .tab-content {
                display: none;
                padding: 40px 30px;
            }
            .tab-content.active { display: block; }
            .form-group {
                margin-bottom: 25px;
            }
            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            .form-group input {
                width: 100%;
                padding: 15px;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            .form-group input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .alert {
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                font-weight: 500;
            }
            .alert-error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .alert-success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .form-footer {
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #e9ecef;
                color: #666;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .password-requirements {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
                line-height: 1.4;
            }
        </style>
    </head>
    <body>
        <div class="auth-container">
            <div class="header">
                <h1>📄 문서 번역 서비스</h1>
                <p>PDF를 한국어로 번역해드립니다</p>
            </div>
            
            <div class="auth-tabs">
                <button class="tab active" onclick="switchTab('login')">로그인</button>
                <button class="tab" onclick="switchTab('register')">회원가입</button>
            </div>
            
            <!-- 로그인 탭 -->
            <div id="login-tab" class="tab-content active">
                <form onsubmit="handleLogin(event)">
                    <div id="login-alerts"></div>
                    
                    <div class="form-group">
                        <label for="login-username">사용자명 또는 이메일</label>
                        <input type="text" id="login-username" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="login-password">비밀번호</label>
                        <input type="password" id="login-password" required>
                    </div>
                    
                    <button type="submit" class="btn" id="login-btn">로그인</button>
                    
                    <div class="loading" id="login-loading">
                        <div class="spinner"></div>
                        <p>로그인 중...</p>
                    </div>
                </form>
                
                <div class="form-footer">
                    <p>계정이 없으신가요? <a href="#" onclick="switchTab('register')" style="color: #667eea; text-decoration: none;">회원가입</a></p>
                </div>
            </div>
            
            <!-- 회원가입 탭 -->
            <div id="register-tab" class="tab-content">
                <form onsubmit="handleRegister(event)">
                    <div id="register-alerts"></div>
                    
                    <div class="form-group">
                        <label for="register-username">사용자명</label>
                        <input type="text" id="register-username" required minlength="3">
                    </div>
                    
                    <div class="form-group">
                        <label for="register-email">이메일</label>
                        <input type="email" id="register-email" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="register-password">비밀번호</label>
                        <input type="password" id="register-password" required minlength="8">
                        <div class="password-requirements">
                            8자 이상, 대문자, 소문자, 숫자를 포함해야 합니다
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="register-password-confirm">비밀번호 확인</label>
                        <input type="password" id="register-password-confirm" required>
                    </div>
                    
                    <button type="submit" class="btn" id="register-btn">회원가입</button>
                    
                    <div class="loading" id="register-loading">
                        <div class="spinner"></div>
                        <p>회원가입 중...</p>
                    </div>
                </form>
                
                <div class="form-footer">
                    <p>이미 계정이 있으신가요? <a href="#" onclick="switchTab('login')" style="color: #667eea; text-decoration: none;">로그인</a></p>
                </div>
            </div>
        </div>
        
        <script>
            function switchTab(tabName) {
                // 모든 탭 비활성화
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // 선택한 탭 활성화
                document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
                document.getElementById(`${tabName}-tab`).classList.add('active');
                
                // 알림 메시지 초기화
                clearAlerts();
            }
            
            function clearAlerts() {
                document.getElementById('login-alerts').innerHTML = '';
                document.getElementById('register-alerts').innerHTML = '';
            }
            
            function showAlert(containerId, message, type = 'error') {
                const container = document.getElementById(containerId);
                container.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
            }
            
            function setLoading(formType, isLoading) {
                const btn = document.getElementById(`${formType}-btn`);
                const loading = document.getElementById(`${formType}-loading`);
                
                btn.disabled = isLoading;
                loading.style.display = isLoading ? 'block' : 'none';
            }
            
            async function handleLogin(event) {
                event.preventDefault();
                
                const username = document.getElementById('login-username').value;
                const password = document.getElementById('login-password').value;
                
                if (!username || !password) {
                    showAlert('login-alerts', '모든 필드를 입력해주세요.');
                    return;
                }
                
                setLoading('login', true);
                clearAlerts();
                
                try {
                    const response = await fetch('/api/v1/auth/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ username, password })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok && result.success) {
                        showAlert('login-alerts', '로그인 성공! 페이지를 새로고침합니다.', 'success');
                        setTimeout(() => {
                            window.location.href = '/';
                        }, 1000);
                    } else {
                        showAlert('login-alerts', result.detail || result.message || '로그인에 실패했습니다.');
                    }
                } catch (error) {
                    showAlert('login-alerts', '서버 연결에 실패했습니다. 나중에 다시 시도해주세요.');
                    console.error('Login error:', error);
                } finally {
                    setLoading('login', false);
                }
            }
            
            async function handleRegister(event) {
                event.preventDefault();
                
                const username = document.getElementById('register-username').value;
                const email = document.getElementById('register-email').value;
                const password = document.getElementById('register-password').value;
                const passwordConfirm = document.getElementById('register-password-confirm').value;
                
                // 클라이언트 사이드 유효성 검증
                if (!username || !email || !password || !passwordConfirm) {
                    showAlert('register-alerts', '모든 필드를 입력해주세요.');
                    return;
                }
                
                if (username.length < 3) {
                    showAlert('register-alerts', '사용자명은 3자 이상이어야 합니다.');
                    return;
                }
                
                if (password !== passwordConfirm) {
                    showAlert('register-alerts', '비밀번호가 일치하지 않습니다.');
                    return;
                }
                
                // 비밀번호 강도 검증
                const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/;
                if (!passwordRegex.test(password)) {
                    showAlert('register-alerts', '비밀번호는 8자 이상, 대소문자와 숫자를 포함해야 합니다.');
                    return;
                }
                
                setLoading('register', true);
                clearAlerts();
                
                try {
                    const response = await fetch('/api/v1/auth/register', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ username, email, password })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok && result.success) {
                        showAlert('register-alerts', '회원가입이 완료되었습니다! 로그인해주세요.', 'success');
                        setTimeout(() => {
                            switchTab('login');
                            document.getElementById('login-username').value = username;
                        }, 1500);
                    } else {
                        showAlert('register-alerts', result.detail || result.message || '회원가입에 실패했습니다.');
                    }
                } catch (error) {
                    showAlert('register-alerts', '서버 연결에 실패했습니다. 나중에 다시 시도해주세요.');
                    console.error('Register error:', error);
                } finally {
                    setLoading('register', false);
                }
            }
            
            // 페이지 로드 시 인증 상태 확인
            window.addEventListener('load', async function() {
                try {
                    const response = await fetch('/api/v1/auth/check');
                    const result = await response.json();
                    
                    if (result.authenticated) {
                        // 이미 로그인된 경우 메인 페이지로 리다이렉트
                        window.location.href = '/';
                    }
                } catch (error) {
                    console.log('Auth check failed:', error);
                }
            });
        </script>
    </body>
    </html>
    """

def get_main_interface_html() -> str:
    """메인 서비스 페이지 HTML (로그인된 사용자용)"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>📄 문서 번역 서비스</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 0;
                background-color: #f8f9fa;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
            }
            .header-left h1 {
                margin: 0;
                font-size: 24px;
            }
            .header-left p {
                margin: 5px 0 0 0;
                opacity: 0.9;
            }
            .header-right {
                display: flex;
                align-items: center;
                gap: 20px;
            }
            .user-info {
                text-align: right;
            }
            .user-name {
                font-weight: bold;
                font-size: 16px;
            }
            .user-email {
                font-size: 14px;
                opacity: 0.8;
            }
            .logout-btn {
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s;
                font-size: 14px;
            }
            .logout-btn:hover {
                background: rgba(255,255,255,0.3);
            }
            .upload-area { 
                border: 2px dashed #ccc; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 10px;
                background: white;
                transition: all 0.3s;
            }
            .upload-area:hover { 
                border-color: #667eea; 
                background-color: #f8f9ff;
            }
            .upload-area.dragover {
                border-color: #667eea;
                background-color: #e8f0fe;
            }
            .progress { 
                width: 100%; 
                height: 25px; 
                background: #f0f0f0; 
                border-radius: 12px; 
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-bar { 
                height: 100%; 
                background: linear-gradient(90deg, #4CAF50, #45a049); 
                transition: width 0.3s; 
                border-radius: 12px;
            }
            .tabs {
                display: flex;
                margin: 20px 0;
                border-bottom: 2px solid #e9ecef;
            }
            .tab {
                padding: 15px 25px;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                font-weight: 500;
                transition: all 0.3s;
            }
            .tab.active {
                border-bottom-color: #667eea;
                color: #667eea;
                background-color: #f8f9ff;
            }
            .tab:hover {
                background-color: #f1f3f4;
            }
            .tab-content {
                display: none;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .tab-content.active {
                display: block;
            }
            .upload-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background: white;
                transition: all 0.3s;
            }
            .upload-item:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }
            .upload-info {
                flex: 1;
            }
            .upload-filename {
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            .upload-meta {
                font-size: 12px;
                color: #666;
                margin-bottom: 8px;
            }
            .upload-status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .status-completed { background: #d4edda; color: #155724; }
            .status-running { background: #d1ecf1; color: #0c5460; }
            .status-failed { background: #f8d7da; color: #721c24; }
            .status-created { background: #fff3cd; color: #856404; }
            .upload-actions {
                display: flex;
                gap: 8px;
            }
            .btn {
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.3s;
                text-decoration: none;
                display: inline-block;
            }
            .btn-primary { background: #667eea; color: white; }
            .btn-primary:hover { background: #5a6fd8; }
            .btn-success { background: #4CAF50; color: white; }
            .btn-success:hover { background: #45a049; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-danger:hover { background: #c82333; }
            .btn-secondary { background: #6c757d; color: white; }
            .btn-secondary:hover { background: #5a6268; }
            .statistics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
            }
            .empty-state {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            .upload-progress {
                width: 100%;
                height: 8px;
                background: #f0f0f0;
                border-radius: 4px;
                margin-top: 5px;
            }
            .upload-progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 4px;
                transition: width 0.3s;
            }
            .error { color: #dc3545; }
            .success { color: #4CAF50; }
            .info { color: #17a2b8; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-left">
                <h1>📄 문서 번역 서비스</h1>
                <p>PDF 파일을 업로드하면 한국어로 번역해드립니다</p>
            </div>
            <div class="header-right">
                <div class="user-info">
                    <div class="user-name" id="user-name">사용자</div>
                    <div class="user-email" id="user-email">user@example.com</div>
                </div>
                <button class="logout-btn" onclick="logout()">로그아웃</button>
            </div>
        </div>
        
        <!-- 탭 네비게이션 -->
        <div class="tabs">
            <div class="tab active" onclick="switchTab('upload')">🔄 새 번역</div>
            <div class="tab" onclick="switchTab('history')">📋 내 기록</div>
            <div class="tab" onclick="switchTab('status')">📊 현재 작업</div>
        </div>
        
        <!-- 새 번역 탭 -->
        <div id="upload-tab" class="tab-content active">
            <div class="upload-area" onclick="document.getElementById('file-input').click()" 
                ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                <input type="file" id="file-input" accept=".pdf" style="display: none;" onchange="uploadFile()">
                <p>📁 클릭하여 PDF 파일을 선택하거나 여기에 끌어다 놓으세요</p>
                <p style="color: #666; font-size: 0.9em;">지원 형식: PDF | 최대 크기: 100MB</p>
            </div>
            
            <div id="upload-status"></div>
        </div>
        
        <!-- 내 기록 탭 -->
        <div id="history-tab" class="tab-content">
            <div class="statistics" id="statistics">
                <!-- 통계 카드들이 여기에 동적으로 추가됩니다 -->
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h3>📋 업로드 기록</h3>
                <button class="btn btn-secondary" onclick="refreshHistory()">🔄 새로고침</button>
            </div>
            
            <div id="history-list">
                <div class="empty-state">
                    <p>아직 업로드 기록이 없습니다.</p>
                    <p>첫 번째 PDF 파일을 업로드해보세요!</p>
                </div>
            </div>
        </div>
        
        <!-- 현재 작업 탭 -->
        <div id="status-tab" class="tab-content">
            <h3>📊 실행 중인 작업</h3>
            <div id="current-workflows"></div>
        </div>
        
        <script>
            let currentTab = 'upload';
            let currentWorkflowId = null;
            let currentUser = null;
            
            // 페이지 로드 시 사용자 정보 로드
            window.addEventListener('load', async function() {
                await loadUserInfo();
                await checkAuthStatus();
            });
            
            async function loadUserInfo() {
                try {
                    const response = await fetch('/api/v1/auth/me');
                    if (response.ok) {
                        currentUser = await response.json();
                        document.getElementById('user-name').textContent = currentUser.username;
                        document.getElementById('user-email').textContent = currentUser.email;
                    }
                } catch (error) {
                    console.error('Failed to load user info:', error);
                }
            }
            
            async function checkAuthStatus() {
                try {
                    const response = await fetch('/api/v1/auth/check');
                    const result = await response.json();
                    
                    if (!result.authenticated) {
                        // 로그인되지 않은 경우 로그인 페이지로 리다이렉트
                        window.location.href = '/login';
                    }
                } catch (error) {
                    console.error('Auth check failed:', error);
                    window.location.href = '/login';
                }
            }
            
            async function logout() {
                try {
                    const response = await fetch('/api/v1/auth/logout', { method: 'POST' });
                    if (response.ok) {
                        window.location.href = '/login';
                    }
                } catch (error) {
                    console.error('Logout failed:', error);
                    // 실패해도 로그인 페이지로 이동
                    window.location.href = '/login';
                }
            }
            
            // 탭 전환
            function switchTab(tabName) {
                // 모든 탭 비활성화
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // 선택한 탭 활성화
                document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
                document.getElementById(`${tabName}-tab`).classList.add('active');
                
                currentTab = tabName;
                
                // 탭별 데이터 로드
                if (tabName === 'history') {
                    refreshHistory();
                } else if (tabName === 'status') {
                    refreshCurrentWorkflows();
                }
            }
            
            // 드래그 앤 드롭 처리
            function handleDragOver(e) {
                e.preventDefault();
                e.currentTarget.classList.add('dragover');
            }
            
            function handleDragLeave(e) {
                e.preventDefault();
                e.currentTarget.classList.remove('dragover');
            }
            
            function handleDrop(e) {
                e.preventDefault();
                e.currentTarget.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const fileInput = document.getElementById('file-input');
                    fileInput.files = files;
                    uploadFile();
                }
            }
            
            // 파일 업로드
            async function uploadFile() {
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    alert('PDF 파일만 업로드 가능합니다.');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('upload-status').innerHTML = `
                    <div class="upload-item">
                        <div class="upload-info">
                            <div class="upload-filename">${file.name}</div>
                            <div class="upload-meta">크기: ${formatFileSize(file.size)} | 업로드 중...</div>
                            <div class="progress">
                                <div class="progress-bar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                `;
                
                try {
                    const response = await fetch('/api/v1/documents/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentWorkflowId = result.workflow_id;
                        document.getElementById('upload-status').innerHTML = `
                            <div class="upload-item">
                                <div class="upload-info">
                                    <div class="upload-filename">✅ ${file.name}</div>
                                    <div class="upload-meta">워크플로우 ID: ${result.workflow_id}</div>
                                    <span class="upload-status status-created">업로드 완료</span>
                                </div>
                            </div>
                            <p class="success">✅ 업로드 성공! 번역을 시작합니다.</p>
                        `;
                        
                        // 진행 상황 모니터링 시작
                        monitorProgress(result.workflow_id);
                        
                        // 기록 탭 자동 새로고침
                        if (currentTab === 'history') {
                            setTimeout(refreshHistory, 1000);
                        }
                    } else {
                        throw new Error(result.detail);
                    }
                } catch (error) {
                    document.getElementById('upload-status').innerHTML = `
                        <p class="error">❌ 업로드 실패: ${error.message}</p>
                    `;
                }
            }
            
            // 진행 상황 모니터링
            async function monitorProgress(workflowId) {
                const interval = setInterval(async () => {
                    try {
                        const response = await fetch(`/api/v1/workflows/${workflowId}`);
                        const workflow = await response.json();
                        
                        if (response.ok) {
                            updateProgressDisplay(workflow);
                            
                            if (workflow.status === 'COMPLETED') {
                                clearInterval(interval);
                                showDownloadLink(workflowId);
                            } else if (workflow.status === 'FAILED') {
                                clearInterval(interval);
                                showError(workflow.error_info);
                            }
                        }
                    } catch (error) {
                        console.error('Progress monitoring failed:', error);
                    }
                }, 2000);
            }
            
            // 진행 상황 업데이트
            function updateProgressDisplay(workflow) {
                const statusDiv = document.getElementById('upload-status');
                
                let detailsHtml = '';
                if (workflow.current_action) {
                    detailsHtml += `<p>📍 현재 작업: ${workflow.current_action}</p>`;
                }
                
                if (workflow.total_pages > 0) {
                    if (workflow.current_stage === 'LAYOUT_ANALYSIS') {
                        detailsHtml += `<p>📄 문서 분석: ${workflow.total_pages}개 페이지</p>`;
                    } else if (workflow.current_stage === 'TRANSLATION' && workflow.current_page > 0) {
                        detailsHtml += `<p>📄 번역 진행: ${workflow.current_page} / ${workflow.total_pages} 페이지</p>`;
                    }
                }
                
                const progressHtml = `
                    <div class="upload-item">
                        <div class="upload-info">
                            <div class="upload-filename">📋 상태: ${getStatusText(workflow.status)}</div>
                            <div class="upload-meta">🔄 단계: ${getStageText(workflow.current_stage)}</div>
                            ${detailsHtml}
                            <div class="progress">
                                <div class="progress-bar" style="width: ${workflow.progress_percentage}%"></div>
                            </div>
                            <p>전체 진행률: ${workflow.progress_percentage}%</p>
                        </div>
                    </div>
                `;
                
                if (statusDiv.innerHTML.includes('워크플로우 ID')) {
                    const parts = statusDiv.innerHTML.split('<p>워크플로우 ID:');
                    statusDiv.innerHTML = progressHtml + '<p>워크플로우 ID:' + parts[1];
                } else {
                    statusDiv.innerHTML = progressHtml;
                }
            }
            
            // 다운로드 링크 표시
            function showDownloadLink(workflowId) {
                const statusDiv = document.getElementById('upload-status');
                // 이미 동일 워크플로우의 다운로드 블록이 존재하면 중복 추가 방지
                if (statusDiv.querySelector(`[data-download-for="${workflowId}"]`)) {
                    return;
                }
                const block = document.createElement('div');
                block.className = 'upload-item';
                block.setAttribute('data-download-for', workflowId);
                block.innerHTML = `
                        <div class="upload-info">
                            <span class="upload-status status-completed">✅ 번역 완료!</span>
                        </div>
                        <div class="upload-actions">
                            <button class="btn btn-success" onclick="downloadResult('${workflowId}')">📥 결과 다운로드</button>
                        </div>
                `;
                statusDiv.appendChild(block);
            }
            
            // 에러 표시
            function showError(errorInfo) {
                const statusDiv = document.getElementById('upload-status');
                statusDiv.innerHTML += `
                    <p class="error">❌ 번역 실패: ${errorInfo ? errorInfo.message : '알 수 없는 오류'}</p>
                `;
            }
            
            // 내 기록 새로고침
            async function refreshHistory() {
                try {
                    const response = await fetch('/api/v1/my-uploads');
                    const data = await response.json();
                    
                    // 통계 표시
                    const stats = data.statistics || {};
                    displayStatistics(stats);
                    
                    // 업로드 목록 표시
                    const uploads = data.uploads || [];
                    displayUploadHistory(uploads);
                    
                } catch (error) {
                    console.error('히스토리 조회 실패:', error);
                    document.getElementById('history-list').innerHTML = `
                        <p class="error">❌ 기록을 불러올 수 없습니다: ${error.message}</p>
                    `;
                }
            }
            
            // 통계 표시
            function displayStatistics(stats) {
                const statsDiv = document.getElementById('statistics');
                statsDiv.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-number">${stats.total_uploads || 0}</div>
                        <div class="stat-label">총 업로드</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats.completed || 0}</div>
                        <div class="stat-label">완료</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats.processing || 0}</div>
                        <div class="stat-label">처리중</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${formatFileSize(stats.total_files_size || 0)}</div>
                        <div class="stat-label">총 용량</div>
                    </div>
                `;
            }
            
            // 업로드 기록 표시
            function displayUploadHistory(uploads) {
                const historyDiv = document.getElementById('history-list');
                
                if (uploads.length === 0) {
                    historyDiv.innerHTML = `
                        <div class="empty-state">
                            <p>아직 업로드 기록이 없습니다.</p>
                            <button class="btn btn-primary" onclick="switchTab('upload')">첫 번째 파일 업로드하기</button>
                        </div>
                    `;
                    return;
                }
                
                historyDiv.innerHTML = uploads.map(upload => `
                    <div class="upload-item">
                        <div class="upload-info">
                            <div class="upload-filename">📄 ${upload.original_filename}</div>
                            <div class="upload-meta">
                                크기: ${formatFileSize(upload.file_size)} | 
                                업로드: ${formatDateTime(upload.upload_time)}
                                ${upload.processing_time ? ` | 처리시간: ${upload.processing_time}초` : ''}
                            </div>
                            <span class="upload-status status-${upload.status.toLowerCase()}">${getStatusText(upload.status)}</span>
                            ${upload.progress < 100 && upload.status === 'RUNNING' ? `
                                <div class="upload-progress">
                                    <div class="upload-progress-bar" style="width: ${upload.progress}%"></div>
                                </div>
                            ` : ''}
                            ${upload.error_message ? `<p class="error">❌ ${upload.error_message}</p>` : ''}
                        </div>
                        <div class="upload-actions">
                            ${upload.status === 'COMPLETED' ? 
                                `<button class="btn btn-success" onclick="downloadUploadResult('${upload.workflow_id}')">다운로드</button>` : 
                                ''}
                            <button class="btn btn-danger" onclick="deleteUpload('${upload.workflow_id}')">삭제</button>
                        </div>
                    </div>
                `).join('');
            }
            
            // 현재 작업 새로고침
            async function refreshCurrentWorkflows() {
                try {
                    const response = await fetch('/api/v1/workflows/');
                    const stats = await response.json();
                    
                    document.getElementById('current-workflows').innerHTML = `
                        <div class="statistics">
                            <div class="stat-card">
                                <div class="stat-number">${stats.running}</div>
                                <div class="stat-label">실행중</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${stats.total}</div>
                                <div class="stat-label">전체</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${stats.completed}</div>
                                <div class="stat-label">완료</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${stats.failed}</div>
                                <div class="stat-label">실패</div>
                            </div>
                        </div>
                    `;
                } catch (error) {
                    console.error('현재 작업 조회 실패:', error);
                }
            }
            
            // 결과 다운로드
            async function downloadResult(workflowId) {
                try {
                    const response = await fetch(`/api/v1/workflows/${workflowId}/download`);
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `translation_result_${workflowId}.zip`;
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        alert('다운로드 실패');
                    }
                } catch (error) {
                    alert('다운로드 오류: ' + error.message);
                }
            }
            
            // 업로드 결과 다운로드
            async function downloadUploadResult(workflowId) {
                try {
                    const response = await fetch(`/api/v1/uploads/${workflowId}/download`);
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `translation_result_${workflowId}.zip`;
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        alert('다운로드 실패');
                    }
                } catch (error) {
                    alert('다운로드 오류: ' + error.message);
                }
            }
            
            // 업로드 삭제
            async function deleteUpload(workflowId) {
                if (!confirm('이 업로드 기록을 삭제하시겠습니까?')) return;
                
                try {
                    const response = await fetch(`/api/v1/uploads/${workflowId}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        refreshHistory();
                        alert('업로드 기록이 삭제되었습니다.');
                    } else {
                        alert('삭제 실패');
                    }
                } catch (error) {
                    alert('삭제 오류: ' + error.message);
                }
            }
            
            // 유틸리티 함수들
            function getStatusText(status) {
                const statusMap = {
                    'CREATED': '생성됨',
                    'RUNNING': '실행중',
                    'COMPLETED': '완료',
                    'FAILED': '실패',
                    'PAUSED': '일시정지'
                };
                return statusMap[status] || status;
            }
            
            function getStageText(stage) {
                const stageMap = {
                    'LAYOUT_ANALYSIS': '레이아웃 분석',
                    'TRANSLATION': '번역',
                    'COMPLETION': '완료'
                };
                return stageMap[stage] || stage;
            }
            
            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            function formatDateTime(dateTimeStr) {
                const date = new Date(dateTimeStr);
                return date.toLocaleString('ko-KR');
            }
            
            // 자동 새로고침 설정
            setInterval(() => {
                if (currentTab === 'history') {
                    refreshHistory();
                } else if (currentTab === 'status') {
                    refreshCurrentWorkflows();
                }
            }, 10000); // 10초마다
        </script>
    </body>
    </html>
    """

def get_web_interface_html() -> str:
    """웹 인터페이스 HTML 생성"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>📄 문서 번역 서비스</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #f8f9fa;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
            }
            .upload-area { 
                border: 2px dashed #ccc; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 10px;
                background: white;
                transition: all 0.3s;
            }
            .upload-area:hover { 
                border-color: #667eea; 
                background-color: #f8f9ff;
            }
            .upload-area.dragover {
                border-color: #667eea;
                background-color: #e8f0fe;
            }
            .progress { 
                width: 100%; 
                height: 25px; 
                background: #f0f0f0; 
                border-radius: 12px; 
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-bar { 
                height: 100%; 
                background: linear-gradient(90deg, #4CAF50, #45a049); 
                transition: width 0.3s; 
                border-radius: 12px;
            }
            .tabs {
                display: flex;
                margin: 20px 0;
                border-bottom: 2px solid #e9ecef;
            }
            .tab {
                padding: 15px 25px;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                font-weight: 500;
                transition: all 0.3s;
            }
            .tab.active {
                border-bottom-color: #667eea;
                color: #667eea;
                background-color: #f8f9ff;
            }
            .tab:hover {
                background-color: #f1f3f4;
            }
            .tab-content {
                display: none;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .tab-content.active {
                display: block;
            }
            .upload-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background: white;
                transition: all 0.3s;
            }
            .upload-item:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }
            .upload-info {
                flex: 1;
            }
            .upload-filename {
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            .upload-meta {
                font-size: 12px;
                color: #666;
                margin-bottom: 8px;
            }
            .upload-status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .status-completed { background: #d4edda; color: #155724; }
            .status-running { background: #d1ecf1; color: #0c5460; }
            .status-failed { background: #f8d7da; color: #721c24; }
            .status-created { background: #fff3cd; color: #856404; }
            .upload-actions {
                display: flex;
                gap: 8px;
            }
            .btn {
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.3s;
                text-decoration: none;
                display: inline-block;
            }
            .btn-primary { background: #667eea; color: white; }
            .btn-primary:hover { background: #5a6fd8; }
            .btn-success { background: #4CAF50; color: white; }
            .btn-success:hover { background: #45a049; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-danger:hover { background: #c82333; }
            .btn-secondary { background: #6c757d; color: white; }
            .btn-secondary:hover { background: #5a6268; }
            .statistics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
            }
            .empty-state {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            .empty-state img {
                width: 100px;
                opacity: 0.5;
                margin-bottom: 20px;
            }
            .upload-progress {
                width: 100%;
                height: 8px;
                background: #f0f0f0;
                border-radius: 4px;
                margin-top: 5px;
            }
            .upload-progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 4px;
                transition: width 0.3s;
            }
            .error { color: #dc3545; }
            .success { color: #4CAF50; }
            .info { color: #17a2b8; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📄 문서 번역 서비스</h1>
            <p>PDF 파일을 업로드하면 한국어로 번역해드립니다</p>
        </div>
        
        <!-- 탭 네비게이션 -->
        <div class="tabs">
            <div class="tab active" onclick="switchTab('upload')">🔄 새 번역</div>
            <div class="tab" onclick="switchTab('history')">📋 내 기록</div>
            <div class="tab" onclick="switchTab('status')">📊 현재 작업</div>
        </div>
        
        <!-- 새 번역 탭 -->
        <div id="upload-tab" class="tab-content active">
            <div class="upload-area" onclick="document.getElementById('file-input').click()" 
                ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                <input type="file" id="file-input" accept=".pdf" style="display: none;" onchange="uploadFile()">
                <p>📁 클릭하여 PDF 파일을 선택하거나 여기에 끌어다 놓으세요</p>
                <p style="color: #666; font-size: 0.9em;">지원 형식: PDF | 최대 크기: 100MB</p>
            </div>
            
            <div id="upload-status"></div>
        </div>
        
        <!-- 내 기록 탭 -->
        <div id="history-tab" class="tab-content">
            <div class="statistics" id="statistics">
                <!-- 통계 카드들이 여기에 동적으로 추가됩니다 -->
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h3>📋 업로드 기록</h3>
                <button class="btn btn-secondary" onclick="refreshHistory()">🔄 새로고침</button>
            </div>
            
            <div id="history-list">
                <div class="empty-state">
                    <p>아직 업로드 기록이 없습니다.</p>
                    <p>첫 번째 PDF 파일을 업로드해보세요!</p>
                </div>
            </div>
        </div>
        
        <!-- 현재 작업 탭 -->
        <div id="status-tab" class="tab-content">
            <h3>📊 실행 중인 작업</h3>
            <div id="current-workflows"></div>
        </div>
        
        <script>
            let currentTab = 'upload';
            let currentWorkflowId = null;
            
            // 탭 전환
            function switchTab(tabName) {
                // 모든 탭 비활성화
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // 선택한 탭 활성화
                document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
                document.getElementById(`${tabName}-tab`).classList.add('active');
                
                currentTab = tabName;
                
                // 탭별 데이터 로드
                if (tabName === 'history') {
                    refreshHistory();
                } else if (tabName === 'status') {
                    refreshCurrentWorkflows();
                }
            }
            
            // 드래그 앤 드롭 처리
            function handleDragOver(e) {
                e.preventDefault();
                e.currentTarget.classList.add('dragover');
            }
            
            function handleDragLeave(e) {
                e.preventDefault();
                e.currentTarget.classList.remove('dragover');
            }
            
            function handleDrop(e) {
                e.preventDefault();
                e.currentTarget.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const fileInput = document.getElementById('file-input');
                    fileInput.files = files;
                    uploadFile();
                }
            }
            
            // 파일 업로드
            async function uploadFile() {
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    alert('PDF 파일만 업로드 가능합니다.');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('upload-status').innerHTML = `
                    <div class="upload-item">
                        <div class="upload-info">
                            <div class="upload-filename">${file.name}</div>
                            <div class="upload-meta">크기: ${formatFileSize(file.size)} | 업로드 중...</div>
                            <div class="progress">
                                <div class="progress-bar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                `;
                
                try {
                    const response = await fetch('/api/v1/documents/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentWorkflowId = result.workflow_id;
                        document.getElementById('upload-status').innerHTML = `
                            <div class="upload-item">
                                <div class="upload-info">
                                    <div class="upload-filename">✅ ${file.name}</div>
                                    <div class="upload-meta">업로드 ID: ${result.upload_id || result.workflow_id}</div>
                                    <span class="upload-status status-created">업로드 완료</span>
                                </div>
                            </div>
                            <p class="success">✅ 업로드 성공! 번역을 시작합니다.</p>
                        `;
                        
                        // 진행 상황 모니터링 시작
                        monitorProgress(result.workflow_id);
                        
                        // 기록 탭 자동 새로고침
                        if (currentTab === 'history') {
                            setTimeout(refreshHistory, 1000);
                        }
                    } else {
                        throw new Error(result.detail);
                    }
                } catch (error) {
                    document.getElementById('upload-status').innerHTML = `
                        <p class="error">❌ 업로드 실패: ${error.message}</p>
                    `;
                }
            }
            
            // 진행 상황 모니터링
            async function monitorProgress(workflowId) {
                const interval = setInterval(async () => {
                    try {
                        const response = await fetch(`/api/v1/workflows/${workflowId}`);
                        const workflow = await response.json();
                        
                        if (response.ok) {
                            updateProgressDisplay(workflow);
                            
                            if (workflow.status === 'COMPLETED') {
                                clearInterval(interval);
                                showDownloadLink(workflowId);
                            } else if (workflow.status === 'FAILED') {
                                clearInterval(interval);
                                showError(workflow.error_info);
                            }
                        }
                    } catch (error) {
                        console.error('Progress monitoring failed:', error);
                    }
                }, 2000);
            }
            
            // 진행 상황 업데이트
            function updateProgressDisplay(workflow) {
                const statusDiv = document.getElementById('upload-status');
                
                let detailsHtml = '';
                if (workflow.current_action) {
                    detailsHtml += `<p>📍 현재 작업: ${workflow.current_action}</p>`;
                }
                
                if (workflow.total_pages > 0) {
                    if (workflow.current_stage === 'LAYOUT_ANALYSIS') {
                        detailsHtml += `<p>📄 문서 분석: ${workflow.total_pages}개 페이지</p>`;
                    } else if (workflow.current_stage === 'TRANSLATION' && workflow.current_page > 0) {
                        detailsHtml += `<p>📄 번역 진행: ${workflow.current_page} / ${workflow.total_pages} 페이지</p>`;
                    }
                }
                
                const progressHtml = `
                    <div class="upload-item">
                        <div class="upload-info">
                            <div class="upload-filename">📋 상태: ${getStatusText(workflow.status)}</div>
                            <div class="upload-meta">🔄 단계: ${getStageText(workflow.current_stage)}</div>
                            ${detailsHtml}
                            <div class="progress">
                                <div class="progress-bar" style="width: ${workflow.progress_percentage}%"></div>
                            </div>
                            <p>전체 진행률: ${workflow.progress_percentage}%</p>
                        </div>
                    </div>
                `;
                
                if (statusDiv.innerHTML.includes('워크플로우 ID')) {
                    const parts = statusDiv.innerHTML.split('<p>워크플로우 ID:');
                    statusDiv.innerHTML = progressHtml + '<p>워크플로우 ID:' + parts[1];
                } else {
                    statusDiv.innerHTML = progressHtml;
                }
            }
            
            // 다운로드 링크 표시
            function showDownloadLink(workflowId) {
                const statusDiv = document.getElementById('upload-status');
                statusDiv.innerHTML += `
                    <div class="upload-item">
                        <div class="upload-info">
                            <span class="upload-status status-completed">✅ 번역 완료!</span>
                        </div>
                        <div class="upload-actions">
                            <button class="btn btn-success" onclick="downloadResult('${workflowId}')">📥 결과 다운로드</button>
                        </div>
                    </div>
                `;
            }
            
            // 에러 표시
            function showError(errorInfo) {
                const statusDiv = document.getElementById('upload-status');
                statusDiv.innerHTML += `
                    <p class="error">❌ 번역 실패: ${errorInfo ? errorInfo.message : '알 수 없는 오류'}</p>
                `;
            }
            
            // 내 기록 새로고침
            async function refreshHistory() {
                try {
                    const response = await fetch('/api/v1/my-uploads');
                    const data = await response.json();
                    
                    // 통계 표시 (방어 코드)
                    const stats = data && data.statistics ? data.statistics : {};
                    displayStatistics(stats);
                    
                    // 업로드 목록 표시 (방어 코드)
                    const uploads = Array.isArray(data && data.uploads) ? data.uploads : [];
                    displayUploadHistory(uploads);
                    
                } catch (error) {
                    console.error('히스토리 조회 실패:', error);
                    document.getElementById('history-list').innerHTML = `
                        <p class="error">❌ 기록을 불러올 수 없습니다: ${error.message}</p>
                    `;
                }
            }
            
            // 통계 표시
            function displayStatistics(stats) {
                stats = stats || {};
                const statsDiv = document.getElementById('statistics');
                statsDiv.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-number">${stats.total_uploads || 0}</div>
                        <div class="stat-label">총 업로드</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats.completed || 0}</div>
                        <div class="stat-label">완료</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats.processing || 0}</div>
                        <div class="stat-label">처리중</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${formatFileSize(stats.total_files_size || 0)}</div>
                        <div class="stat-label">총 용량</div>
                    </div>
                `;
            }
            
            // 업로드 기록 표시
            function displayUploadHistory(uploads) {
                const historyDiv = document.getElementById('history-list');
                
                if (uploads.length === 0) {
                    historyDiv.innerHTML = `
                        <div class="empty-state">
                            <p>아직 업로드 기록이 없습니다.</p>
                            <button class="btn btn-primary" onclick="switchTab('upload')">첫 번째 파일 업로드하기</button>
                        </div>
                    `;
                    return;
                }
                
                historyDiv.innerHTML = uploads.map(upload => `
                    <div class="upload-item">
                        <div class="upload-info">
                            <div class="upload-filename">📄 ${upload.original_filename}</div>
                            <div class="upload-meta">
                                크기: ${formatFileSize(upload.file_size)} | 
                                업로드: ${formatDateTime(upload.upload_time)}
                                ${upload.processing_time ? ` | 처리시간: ${upload.processing_time}초` : ''}
                            </div>
                            <span class="upload-status status-${upload.status.toLowerCase()}">${getStatusText(upload.status)}</span>
                            ${upload.progress < 100 && upload.status === 'RUNNING' ? `
                                <div class="upload-progress">
                                    <div class="upload-progress-bar" style="width: ${upload.progress}%"></div>
                                </div>
                            ` : ''}
                            ${upload.error_message ? `<p class="error">❌ ${upload.error_message}</p>` : ''}
                        </div>
                        <div class="upload-actions">
                            <button class="btn btn-secondary" onclick="viewUploadDetail('${upload.id}')">상세보기</button>
                            ${upload.status === 'COMPLETED' ? 
                                `<button class="btn btn-success" onclick="downloadUploadResult('${upload.id}')">다운로드</button>` : 
                                ''}
                            <button class="btn btn-danger" onclick="deleteUpload('${upload.id}')">삭제</button>
                        </div>
                    </div>
                `).join('');
            }
            
            // 현재 작업 새로고침
            async function refreshCurrentWorkflows() {
                try {
                    const response = await fetch('/api/v1/workflows/');
                    const stats = await response.json();
                    
                    document.getElementById('current-workflows').innerHTML = `
                        <div class="statistics">
                            <div class="stat-card">
                                <div class="stat-number">${stats.running}</div>
                                <div class="stat-label">실행중</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${stats.total}</div>
                                <div class="stat-label">전체</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${stats.completed}</div>
                                <div class="stat-label">완료</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${stats.failed}</div>
                                <div class="stat-label">실패</div>
                            </div>
                        </div>
                    `;
                } catch (error) {
                    console.error('현재 작업 조회 실패:', error);
                }
            }
            
            // 결과 다운로드
            async function downloadResult(workflowId) {
                try {
                    const response = await fetch(`/api/v1/workflows/${workflowId}/download`);
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `translation_result_${workflowId}.zip`;
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        alert('다운로드 실패');
                    }
                } catch (error) {
                    alert('다운로드 오류: ' + error.message);
                }
            }
            
            // 업로드 결과 다운로드
            async function downloadUploadResult(uploadId) {
                try {
                    const response = await fetch(`/api/v1/uploads/${uploadId}/download`);
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `translation_result_${uploadId}.zip`;
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        alert('다운로드 실패');
                    }
                } catch (error) {
                    alert('다운로드 오류: ' + error.message);
                }
            }
            
            // 업로드 삭제
            async function deleteUpload(uploadId) {
                if (!confirm('이 업로드 기록을 삭제하시겠습니까?')) return;
                
                try {
                    const response = await fetch(`/api/v1/uploads/${uploadId}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        refreshHistory();
                        alert('업로드 기록이 삭제되었습니다.');
                    } else {
                        alert('삭제 실패');
                    }
                } catch (error) {
                    alert('삭제 오류: ' + error.message);
                }
            }
            
            // 업로드 상세보기
            async function viewUploadDetail(uploadId) {
                try {
                    const response = await fetch(`/api/v1/uploads/${uploadId}`);
                    const data = await response.json();
                    
                    alert(`상세 정보:\n업로드 ID: ${data.upload.id}\n파일명: ${data.upload.original_filename}\n상태: ${data.upload.status}\n진행률: ${data.upload.progress}%`);
                } catch (error) {
                    alert('상세 정보 조회 실패: ' + error.message);
                }
            }
            
            // 유틸리티 함수들
            function getStatusText(status) {
                const statusMap = {
                    'CREATED': '생성됨',
                    'RUNNING': '실행중',
                    'COMPLETED': '완료',
                    'FAILED': '실패',
                    'PAUSED': '일시정지'
                };
                return statusMap[status] || status;
            }
            
            function getStageText(stage) {
                const stageMap = {
                    'LAYOUT_ANALYSIS': '레이아웃 분석',
                    'TRANSLATION': '번역',
                    'COMPLETION': '완료'
                };
                return stageMap[stage] || stage;
            }
            
            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            function formatDateTime(dateTimeStr) {
                const date = new Date(dateTimeStr);
                return date.toLocaleString('ko-KR');
            }
            
            // 페이지 로드 시 초기화
            window.addEventListener('load', function() {
                // 자동 새로고침 설정
                setInterval(() => {
                    if (currentTab === 'history') {
                        refreshHistory();
                    } else if (currentTab === 'status') {
                        refreshCurrentWorkflows();
                    }
                }, 10000); // 10초마다 자동 새로고침
            });
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=app_config.api_host,
        port=app_config.api_port,
        reload=app_config.api_debug
    )