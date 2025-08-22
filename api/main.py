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

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from core.workflow_manager import WorkflowManager
from core.error_handler import WorkflowErrorHandler
from core.config import app_config
from core.models import WorkflowStatus, WorkflowStage
# 기존 서비스 파일들 import
from services.layout_analysis_service import LayoutAnalysisService, LayoutAnalysisResult
# 페이지별 레이아웃 분석 서비스 (테스트용)
from services.layout_analysis_service_paged import LayoutAnalysisServicePaged
from services.translation_service import TranslationService, BatchTranslationResult

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


# 문서 처리 클래스
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


# API 엔드포인트들
@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    PDF 문서 업로드 및 번역 시작
    """
    # 파일 유효성 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # 출력 디렉토리 생성
    output_dir = OUTPUT_DIR / f"output_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    # 문서 처리 시작
    try:
        result = await document_processor.process_document(str(file_path), str(output_dir))
        return UploadResponse(**result)
    except Exception as e:
        # 업로드된 파일 정리
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


# 웹 인터페이스
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """
    웹 인터페이스 메인 페이지
    """
    return HTMLResponse(content=get_web_interface_html())


def get_web_interface_html() -> str:
    """웹 인터페이스 HTML 생성"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doc test</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #999; }
            .progress { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
            .progress-bar { height: 100%; background: #4CAF50; transition: width 0.3s; }
            .workflow-item { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .status-running { border-left: 4px solid #2196F3; }
            .status-completed { border-left: 4px solid #4CAF50; }
            .status-failed { border-left: 4px solid #f44336; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #45a049; }
            .error { color: #f44336; }
            .success { color: #4CAF50; }
        </style>
    </head>
    <body>
        <h1>📄 doc 서비스</h1>
        <p>PDF 파일을 업로드하면 한국어로 번역해드립니다.</p>
        
        <div class="upload-area" onclick="document.getElementById('file-input').click()">
            <input type="file" id="file-input" accept=".pdf" style="display: none;" onchange="uploadFile()">
            <p>📁 클릭하여 PDF 파일을 선택하거나 여기에 끌어다 놓으세요</p>
        </div>
        
        <div id="upload-status"></div>
        
        <h2>📊 워크플로우 목록</h2>
        <button onclick="refreshWorkflows()">🔄 새로고침</button>
        <div id="workflows-list"></div>
        
        <script>
            let currentWorkflowId = null;
            
            async function uploadFile() {
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('upload-status').innerHTML = '<p>⏳ 파일 업로드 중...</p>';
                
                try {
                    const response = await fetch('/api/v1/documents/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentWorkflowId = result.workflow_id;
                        document.getElementById('upload-status').innerHTML = 
                            `<p class="success">✅ 업로드 성공! 번역을 시작합니다.</p>
                             <p>워크플로우 ID: ${result.workflow_id}</p>`;
                        
                        // 진행 상황 모니터링 시작
                        monitorProgress(result.workflow_id);
                        refreshWorkflows();
                    } else {
                        throw new Error(result.detail);
                    }
                } catch (error) {
                    document.getElementById('upload-status').innerHTML = 
                        `<p class="error">❌ 업로드 실패: ${error.message}</p>`;
                }
            }
            
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
            
            function updateProgressDisplay(workflow) {
                const statusDiv = document.getElementById('upload-status');
                
                // 상세 진행 정보 생성
                let detailsHtml = '';
                if (workflow.current_action) {
                    detailsHtml += `<p>📍 현재 작업: ${workflow.current_action}</p>`;
                }
                
                // 페이지 정보 표시
                if (workflow.total_pages > 0) {
                    // 레이아웃 분석 단계
                    if (workflow.current_stage === 'LAYOUT_ANALYSIS') {
                        detailsHtml += `<p>📄 문서 페이지 분석 중: ${workflow.total_pages}개 페이지 발견</p>`;
                    }
                    // 번역 단계
                    else if (workflow.current_stage === 'TRANSLATION' && workflow.current_page > 0) {
                        detailsHtml += `<p>📄 번역 진행: ${workflow.current_page} / ${workflow.total_pages} 페이지</p>`;
                    }
                }
                
                // 단계별 상세 정보
                if (workflow.stage_details) {
                    if (workflow.stage_details.current_file) {
                        detailsHtml += `<p>📝 현재 파일: ${workflow.stage_details.current_file}</p>`;
                    }
                    if (workflow.stage_details.translated_files !== undefined && workflow.stage_details.failed_files !== undefined) {
                        detailsHtml += `<p>✅ 번역 완료: ${workflow.stage_details.translated_files}개 | ❌ 실패: ${workflow.stage_details.failed_files}개</p>`;
                    }
                }
                
                const progressHtml = `
                    <div>
                        <p>📋 상태: ${getStatusText(workflow.status)}</p>
                        <p>🔄 단계: ${getStageText(workflow.current_stage)}</p>
                        ${detailsHtml}
                        <div class="progress">
                            <div class="progress-bar" style="width: ${workflow.progress_percentage}%"></div>
                        </div>
                        <p>전체 진행률: ${workflow.progress_percentage}%</p>
                    </div>
                `;
                
                if (statusDiv.innerHTML.includes('워크플로우 ID')) {
                    const parts = statusDiv.innerHTML.split('<p>워크플로우 ID:');
                    statusDiv.innerHTML = progressHtml + '<p>워크플로우 ID:' + parts[1];
                } else {
                    statusDiv.innerHTML = progressHtml;
                }
            }
            
            function showDownloadLink(workflowId) {
                const statusDiv = document.getElementById('upload-status');
                statusDiv.innerHTML += `
                    <p class="success">✅ 번역 완료!</p>
                    <button onclick="downloadResult('${workflowId}')">📥 결과 다운로드</button>
                `;
            }
            
            function showError(errorInfo) {
                const statusDiv = document.getElementById('upload-status');
                statusDiv.innerHTML += `
                    <p class="error">❌ 번역 실패: ${errorInfo ? errorInfo.message : '알 수 없는 오류'}</p>
                `;
            }
            
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
            
            async function refreshWorkflows() {
                try {
                    const response = await fetch('/api/v1/workflows/');
                    const stats = await response.json();
                    
                    document.getElementById('workflows-list').innerHTML = `
                        <div class="workflow-item">
                            <h3>📊 전체 통계</h3>
                            <p>전체: ${stats.total} | 실행중: ${stats.running} | 완료: ${stats.completed} | 실패: ${stats.failed}</p>
                        </div>
                    `;
                } catch (error) {
                    console.error('Failed to refresh workflows:', error);
                }
            }
            
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
            
            // 페이지 로드 시 워크플로우 목록 조회
            refreshWorkflows();
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