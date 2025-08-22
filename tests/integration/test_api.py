# tests/integration/test_api.py
import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io

from api.main import app
from core.models import WorkflowStatus, WorkflowStage


class TestDocumentAPI:
    """문서 번역 API 테스트"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """각 테스트 후 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_upload_document_success(self):
        """문서 업로드 성공 테스트"""
        # Given
        pdf_content = b"fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        
        # When
        with patch('api.main.document_processor.process_document') as mock_process:
            mock_process.return_value = {"workflow_id": "test-123"}
            
            response = self.client.post("/api/v1/documents/upload", files=files)
        
        # Then
        assert response.status_code == 200
        data = response.json()
        assert "workflow_id" in data
        assert data["status"] == "processing"
    
    def test_upload_invalid_file_type(self):
        """잘못된 파일 타입 업로드 테스트"""
        # Given
        txt_content = b"not a pdf"
        files = {"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
        
        # When
        response = self.client.post("/api/v1/documents/upload", files=files)
        
        # Then
        assert response.status_code == 400
        assert "Only PDF files are allowed" in response.json()["detail"]
    
    def test_upload_empty_file(self):
        """빈 파일 업로드 테스트"""
        # Given
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        
        # When
        response = self.client.post("/api/v1/documents/upload", files=files)
        
        # Then
        assert response.status_code == 400
        assert "File is empty" in response.json()["detail"]
    
    def test_get_workflow_status_success(self):
        """워크플로우 상태 조회 성공 테스트"""
        # Given
        workflow_id = "test-123"
        
        # When
        with patch('api.main.workflow_manager.get_workflow') as mock_get:
            mock_workflow = MagicMock()
            mock_workflow.to_dict.return_value = {
                "id": workflow_id,
                "status": "RUNNING",
                "progress_percentage": 50
            }
            mock_get.return_value = mock_workflow
            
            response = self.client.get(f"/api/v1/workflows/{workflow_id}")
        
        # Then
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == workflow_id
        assert data["status"] == "RUNNING"
        assert data["progress_percentage"] == 50
    
    def test_get_workflow_not_found(self):
        """존재하지 않는 워크플로우 조회 테스트"""
        # Given
        workflow_id = "non-existent"
        
        # When
        with patch('api.main.workflow_manager.get_workflow') as mock_get:
            mock_get.side_effect = FileNotFoundError("Workflow not found")
            
            response = self.client.get(f"/api/v1/workflows/{workflow_id}")
        
        # Then
        assert response.status_code == 404
        assert "Workflow not found" in response.json()["detail"]
    
    def test_list_workflows(self):
        """워크플로우 목록 조회 테스트"""
        # When
        with patch('api.main.workflow_manager.get_workflow_statistics') as mock_stats:
            mock_stats.return_value = {
                "total": 5,
                "running": 2,
                "completed": 2,
                "failed": 1
            }
            
            response = self.client.get("/api/v1/workflows/")
        
        # Then
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert data["running"] == 2
    
    def test_download_result_success(self):
        """번역 결과 다운로드 성공 테스트"""
        # Given
        workflow_id = "test-123"
        
        # When
        with patch('api.main.workflow_manager.get_workflow') as mock_get, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_workflow = MagicMock()
            mock_workflow.status = WorkflowStatus.COMPLETED
            mock_workflow.output_directory = "/test/output"
            mock_get.return_value = mock_workflow
            
            mock_exists.return_value = True
            mock_open.return_value.__enter__.return_value.read.return_value = b"translated content"
            
            response = self.client.get(f"/api/v1/workflows/{workflow_id}/download")
        
        # Then
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
    
    def test_download_result_not_completed(self):
        """완료되지 않은 워크플로우 다운로드 시도 테스트"""
        # Given
        workflow_id = "test-123"
        
        # When
        with patch('api.main.workflow_manager.get_workflow') as mock_get:
            mock_workflow = MagicMock()
            mock_workflow.status = WorkflowStatus.RUNNING
            mock_get.return_value = mock_workflow
            
            response = self.client.get(f"/api/v1/workflows/{workflow_id}/download")
        
        # Then
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]
    
    def test_health_check(self):
        """헬스 체크 테스트"""
        # When
        response = self.client.get("/health")
        
        # Then
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_web_interface(self):
        """웹 인터페이스 접근 테스트"""
        # When
        response = self.client.get("/")
        
        # Then
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]