# tests/test_layout_analysis_service.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from services.layout_analysis_service import LayoutAnalysisService, LayoutAnalysisResult

class TestLayoutAnalysisService:
    
    @pytest.fixture
    def service(self):
        """테스트용 서비스 인스턴스"""
        config = {
            'use_gpu': False,  # 테스트에서는 CPU 사용
            'det_limit_side_len': 960,
            'use_table': True
        }
        return LayoutAnalysisService(config)
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """테스트용 가짜 PDF 파일"""
        pdf_path = Path(temp_dir) / "test_document.pdf"
        pdf_path.write_bytes(b"fake pdf content")
        return str(pdf_path)
    
    @pytest.fixture
    def mock_paddle_result(self):
        """PaddleOCR 결과 모킹"""
        return [
            {
                "layout": [
                    {
                        "type": "title",
                        "text": "Sample Document Title",
                        "bbox": [100, 50, 400, 80],
                        "confidence": 0.95
                    },
                    {
                        "type": "text", 
                        "text": "This is sample content from the document.",
                        "bbox": [100, 100, 400, 120],
                        "confidence": 0.88
                    }
                ]
            }
        ]

    def test_init_with_default_config(self):
        """기본 설정으로 초기화 테스트"""
        service = LayoutAnalysisService()
        
        assert service.config == {}
        assert service.use_gpu is True  # 기본값
        assert service.det_limit_side_len == 1920
        assert service.use_table is True
    
    def test_init_with_custom_config(self):
        """커스텀 설정으로 초기화 테스트"""
        config = {
            'use_gpu': False,
            'det_limit_side_len': 1280,
            'use_table': False
        }
        service = LayoutAnalysisService(config)
        
        assert service.config == config
        assert service.use_gpu is False
        assert service.det_limit_side_len == 1280
        assert service.use_table is False

    def test_analyze_document_file_not_found(self, service, temp_dir):
        """존재하지 않는 파일 처리 테스트"""
        nonexistent_file = Path(temp_dir) / "nonexistent.pdf"
        
        result = service.analyze_document(str(nonexistent_file), temp_dir)
        
        assert result.success is False
        assert result.error is not None
        assert "찾을 수 없습니다" in result.error
        assert len(result.pages) == 0
        assert len(result.markdown_files) == 0
        assert result.confidence == 0.0

    @patch('services.layout_analysis_service.PPStructureV3')
    def test_analyze_document_success(self, mock_pp_structure, service, sample_pdf, temp_dir, mock_paddle_result):
        """성공적인 문서 분석 테스트"""
        # PaddleOCR 모킹
        mock_pipeline = Mock()
        mock_pipeline.predict.return_value = mock_paddle_result
        mock_pp_structure.return_value = mock_pipeline
        
        result = service.analyze_document(sample_pdf, temp_dir)
        
        assert result.success is True
        assert result.error is None
        assert len(result.pages) == 1
        assert result.confidence > 0
        assert result.processing_time > 0
        assert result.output_dir == temp_dir
        
        # 마크다운 파일이 생성되었는지 확인
        assert len(result.markdown_files) > 0
        for md_file in result.markdown_files:
            assert Path(md_file).exists()

    @patch('services.layout_analysis_service.PPStructureV3')
    def test_analyze_document_paddle_error(self, mock_pp_structure, service, sample_pdf, temp_dir):
        """PaddleOCR 에러 처리 테스트"""
        # PaddleOCR 에러 시뮬레이션
        mock_pipeline = Mock()
        mock_pipeline.predict.side_effect = RuntimeError("GPU 메모리 부족")
        mock_pp_structure.return_value = mock_pipeline
        
        result = service.analyze_document(sample_pdf, temp_dir)
        
        assert result.success is False
        assert "GPU 메모리 부족" in result.error
        assert len(result.pages) == 0
        assert result.confidence == 0.0

    # GPU 감지 테스트는 이제 _get_pipeline 내부에서 처리됨

    def test_calculate_confidence_empty_result(self, service):
        """빈 결과에 대한 신뢰도 계산 테스트"""
        confidence = service._calculate_confidence([])
        assert confidence == 0.0

    def test_calculate_confidence_with_scores(self, service):
        """신뢰도 점수가 있는 결과 테스트"""
        result_list = [
            {
                "layout": [
                    {"confidence": 0.9},
                    {"confidence": 0.8}
                ]
            },
            {
                "layout": [
                    {"confidence": 0.85}
                ]
            }
        ]
        
        confidence = service._calculate_confidence(result_list)
        expected = (0.9 + 0.8 + 0.85) / 3
        assert confidence == expected

    def test_calculate_confidence_no_scores(self, service):
        """신뢰도 점수가 없는 결과 테스트"""
        result_list = [
            {
                "layout": [
                    {"text": "sample text"}
                ]
            }
        ]
        
        confidence = service._calculate_confidence(result_list)
        assert confidence == 0.8  # 기본값

    @patch('services.layout_analysis_service.PPStructureV3')
    def test_generate_fallback_markdown(self, mock_pp_structure, service, sample_pdf, temp_dir, mock_paddle_result):
        """폴백 마크다운 생성 테스트"""
        # save_structure_res가 없는 상황 시뮬레이션
        with patch('services.layout_analysis_service.save_structure_res', side_effect=ImportError()):
            mock_pipeline = Mock()
            mock_pipeline.predict.return_value = mock_paddle_result
            mock_pp_structure.return_value = mock_pipeline
            
            result = service.analyze_document(sample_pdf, temp_dir)
            
            assert result.success is True
            assert len(result.markdown_files) > 0
            
            # 생성된 마크다운 파일 내용 확인
            md_file = Path(result.markdown_files[0])
            assert md_file.exists()
            content = md_file.read_text(encoding='utf-8')
            assert "Sample Document Title" in content
            assert "This is sample content" in content

    def test_output_dir_auto_generation(self, service, sample_pdf):
        """출력 디렉토리 자동 생성 테스트"""
        with patch('services.layout_analysis_service.PPStructureV3') as mock_pp_structure:
            mock_pipeline = Mock()
            mock_pipeline.predict.return_value = []
            mock_pp_structure.return_value = mock_pipeline
            
            result = service.analyze_document(sample_pdf)  # output_dir 지정하지 않음
            
            assert result.output_dir != ""
            assert Path(result.output_dir).exists()
            # 정리
            shutil.rmtree(result.output_dir, ignore_errors=True)

    @patch('services.layout_analysis_service.PPStructureV3')
    def test_json_output_saved(self, mock_pp_structure, service, sample_pdf, temp_dir, mock_paddle_result):
        """JSON 결과 파일 저장 테스트"""
        mock_pipeline = Mock()
        mock_pipeline.predict.return_value = mock_paddle_result
        mock_pp_structure.return_value = mock_pipeline
        
        result = service.analyze_document(sample_pdf, temp_dir)
        
        # JSON 파일이 생성되었는지 확인
        json_file = Path(temp_dir) / "result.json"
        assert json_file.exists()
        
        # JSON 내용 확인
        with json_file.open('r', encoding='utf-8') as f:
            saved_data = json.load(f)
        assert saved_data == mock_paddle_result


# tests/conftest.py (pytest 설정 파일)
import pytest
import logging

def pytest_configure(config):
    """pytest 설정"""
    logging.basicConfig(level=logging.INFO)


# 실행 예시:
# pytest tests/test_layout_analysis_service.py -v