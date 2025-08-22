# tests/test_translation_service.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from services.translation_service import TranslationService, TranslationResult, BatchTranslationResult

class TestTranslationService:
    
    @pytest.fixture
    def service(self):
        """테스트용 서비스 인스턴스"""
        config = {
            'model_name': 'qwen3:8b',
            'temperature': 0.1,
            'max_retries': 2,
            'ollama_url': 'http://localhost:11434',
            'quality_threshold': 0.6
        }
        return TranslationService(config)
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_markdown(self, temp_dir):
        """테스트용 마크다운 파일"""
        md_content = """# Sample Document

## Article 1
This is a sample English document for testing translation.

### Section 1.1
Some legal text that needs to be translated to Korean.

## Article 2
Additional content with technical terms.
"""
        md_path = Path(temp_dir) / "sample.md"
        md_path.write_text(md_content, encoding='utf-8')
        return str(md_path)
    
    @pytest.fixture
    def multiple_markdown_files(self, temp_dir):
        """테스트용 여러 마크다운 파일"""
        files = []
        for i in range(3):
            md_content = f"""# Document {i+1}

## Article {i+1}
This is document number {i+1} for batch translation testing.
"""
            md_path = Path(temp_dir) / f"doc_{i+1}.md"
            md_path.write_text(md_content, encoding='utf-8')
            files.append(str(md_path))
        return files
    
    @pytest.fixture
    def mock_translation_report(self):
        """번역 리포트 모킹"""
        return {
            'input_file': 'sample.md',
            'output_file': 'sample_korean.md',
            'sections_count': 3,
            'average_confidence': 0.85,
            'total_processing_time': 15.2,
            'model_used': 'qwen3:8b',
            'low_quality_sections': []
        }

    def test_init_with_default_config(self):
        """기본 설정으로 초기화 테스트"""
        service = TranslationService()
        
        assert service.model_name == 'qwen3:8b'
        assert service.temperature == 0.1
        assert service.max_retries == 3
        assert service.quality_threshold == 0.6
    
    def test_init_with_custom_config(self):
        """커스텀 설정으로 초기화 테스트"""
        config = {
            'model_name': 'llama3.1:8b',
            'temperature': 0.2,
            'max_retries': 5,
            'quality_threshold': 0.7
        }
        service = TranslationService(config)
        
        assert service.model_name == 'llama3.1:8b'
        assert service.temperature == 0.2
        assert service.max_retries == 5
        assert service.quality_threshold == 0.7

    def test_translate_document_file_not_found(self, service, temp_dir):
        """존재하지 않는 파일 번역 테스트"""
        nonexistent_file = Path(temp_dir) / "nonexistent.md"
        
        result = service.translate_document(str(nonexistent_file))
        
        assert result.success is False
        assert result.error is not None
        assert "찾을 수 없습니다" in result.error
        assert result.processing_time > 0

    @patch('services.translation_service.QualityContractTranslator')
    def test_translate_document_success(self, mock_translator_class, service, sample_markdown, temp_dir, mock_translation_report):
        """성공적인 문서 번역 테스트"""
        # QualityContractTranslator 모킹
        mock_translator = Mock()
        mock_translator.translate_document_premium.return_value = mock_translation_report
        mock_translator_class.return_value = mock_translator
        
        # 출력 파일 생성 시뮬레이션
        output_file = Path(temp_dir) / "sample_korean.md"
        output_file.write_text("번역된 내용", encoding='utf-8')
        
        result = service.translate_document(sample_markdown, str(output_file))
        
        assert result.success is True
        assert result.error is None
        assert result.input_file == sample_markdown
        assert result.output_file == str(output_file)
        assert result.report == mock_translation_report
        assert result.processing_time > 0
        
        # 번역기가 올바른 인자로 호출되었는지 확인
        mock_translator.translate_document_premium.assert_called_once_with(
            input_file=sample_markdown,
            output_file=str(output_file)
        )

    @patch('services.translation_service.QualityContractTranslator')
    def test_translate_document_low_quality(self, mock_translator_class, service, sample_markdown, temp_dir):
        """낮은 품질의 번역 결과 테스트"""
        # 낮은 신뢰도 리포트
        low_quality_report = {
            'average_confidence': 0.4,  # 임계값(0.6)보다 낮음
            'low_quality_sections': ['section1', 'section2']
        }
        
        mock_translator = Mock()
        mock_translator.translate_document_premium.return_value = low_quality_report
        mock_translator_class.return_value = mock_translator
        
        # 출력 파일 생성
        output_file = Path(temp_dir) / "sample_korean.md"
        output_file.write_text("낮은 품질 번역", encoding='utf-8')
        
        result = service.translate_document(sample_markdown, str(output_file))
        
        assert result.success is False  # 품질이 낮아서 실패로 처리
        assert result.report['average_confidence'] == 0.4

    @patch('services.translation_service.QualityContractTranslator')
    def test_translate_document_translator_error(self, mock_translator_class, service, sample_markdown):
        """번역기 에러 처리 테스트"""
        # 번역기 에러 시뮬레이션
        mock_translator = Mock()
        mock_translator.translate_document_premium.side_effect = RuntimeError("Ollama 서버 연결 실패")
        mock_translator_class.return_value = mock_translator
        
        result = service.translate_document(sample_markdown)
        
        assert result.success is False
        assert "Ollama 서버 연결 실패" in result.error

    @patch('services.translation_service.QualityContractTranslator')
    def test_translate_documents_batch_success(self, mock_translator_class, service, multiple_markdown_files, temp_dir, mock_translation_report):
        """배치 번역 성공 테스트"""
        mock_translator = Mock()
        mock_translator.translate_document_premium.return_value = mock_translation_report
        mock_translator_class.return_value = mock_translator
        
        # 출력 파일들 생성 시뮬레이션
        for i, md_file in enumerate(multiple_markdown_files):
            output_path = Path(md_file).parent / f"doc_{i+1}_korean.md"
            output_path.write_text(f"번역된 문서 {i+1}", encoding='utf-8')
        
        result = service.translate_documents_batch(multiple_markdown_files)
        
        assert result.success is True
        assert result.total_files == 3
        assert result.successful_files == 3
        assert result.failed_files == 0
        assert result.average_confidence == mock_translation_report['average_confidence']
        assert len(result.results) == 3
        assert all(r.success for r in result.results)

    @patch('services.translation_service.QualityContractTranslator')
    def test_translate_documents_batch_partial_failure(self, mock_translator_class, service, multiple_markdown_files, temp_dir):
        """배치 번역 부분 실패 테스트"""
        mock_translator = Mock()
        
        # 첫 번째와 세 번째는 성공, 두 번째는 실패
        def side_effect(*args, **kwargs):
            if 'doc_2' in kwargs['input_file']:
                raise RuntimeError("번역 실패")
            return {'average_confidence': 0.8}
        
        mock_translator.translate_document_premium.side_effect = side_effect
        mock_translator_class.return_value = mock_translator
        
        # 성공한 파일들만 출력 파일 생성
        for i in [0, 2]:  # doc_1, doc_3
            output_path = Path(multiple_markdown_files[i]).parent / f"doc_{i+1}_korean.md"
            output_path.write_text(f"번역된 문서 {i+1}", encoding='utf-8')
        
        result = service.translate_documents_batch(multiple_markdown_files)
        
        assert result.success is False
        assert result.total_files == 3
        assert result.successful_files == 2
        assert result.failed_files == 1
        assert len(result.results) == 3
        
        # 실패한 결과 확인
        failed_result = next(r for r in result.results if not r.success)
        assert 'doc_2' in failed_result.input_file

    @patch('services.translation_service.QualityContractTranslator')
    def test_translate_documents_batch_with_output_dir(self, mock_translator_class, service, multiple_markdown_files, temp_dir, mock_translation_report):
        """출력 디렉토리 지정한 배치 번역 테스트"""
        mock_translator = Mock()
        mock_translator.translate_document_premium.return_value = mock_translation_report
        mock_translator_class.return_value = mock_translator
        
        output_dir = Path(temp_dir) / "translated"
        output_dir.mkdir()
        
        # 출력 파일들 생성
        for i in range(3):
            output_path = output_dir / f"doc_{i+1}_korean.md"
            output_path.write_text(f"번역된 문서 {i+1}", encoding='utf-8')
        
        result = service.translate_documents_batch(multiple_markdown_files, str(output_dir))
        
        assert result.success is True
        assert all(str(output_dir) in r.output_file for r in result.results if r.success)

    def test_retry_failed_translations_no_failures(self, service):
        """실패한 번역이 없는 경우 재시도 테스트"""
        # 모든 번역이 성공한 경우
        successful_results = [
            TranslationResult(True, "file1.md", "file1_korean.md", {'average_confidence': 0.8}, 10.0),
            TranslationResult(True, "file2.md", "file2_korean.md", {'average_confidence': 0.9}, 12.0)
        ]
        
        batch_result = BatchTranslationResult(
            success=True,
            results=successful_results,
            total_files=2,
            successful_files=2,
            failed_files=0,
            average_confidence=0.85,
            total_processing_time=22.0,
            low_quality_files=[]
        )
        
        retry_result = service.retry_failed_translations(batch_result)
        
        assert retry_result == batch_result  # 변경되지 않음

    def test_get_translation_status(self, service, sample_markdown, temp_dir):
        """번역 상태 확인 테스트"""
        # 출력 파일 생성
        output_file = Path(temp_dir) / "sample_korean.md"
        output_file.write_text("번역된 내용", encoding='utf-8')
        
        status = service.get_translation_status([sample_markdown])
        
        file_status = status[sample_markdown]
        assert file_status['input_exists'] is True
        assert file_status['output_exists'] is True
        assert file_status['input_size'] > 0
        assert file_status['output_size'] > 0
        assert file_status['needs_translation'] is False  # 출력 파일이 더 최신

    def test_get_translation_status_needs_translation(self, service, sample_markdown):
        """번역이 필요한 상태 확인 테스트"""
        status = service.get_translation_status([sample_markdown])
        
        file_status = status[sample_markdown]
        assert file_status['input_exists'] is True
        assert file_status['output_exists'] is False
        assert file_status['needs_translation'] is True

    def test_cleanup_temp_files(self, service, temp_dir):
        """임시 파일 정리 테스트"""
        # 임시 파일들 생성
        raw_file = Path(temp_dir) / "sample_raw.md"
        raw_file.write_text("raw content", encoding='utf-8')
        
        cache_file = Path(temp_dir) / "translation_cache.json"
        cache_file.write_text('{"cache": "data"}', encoding='utf-8')
        
        normal_file = Path(temp_dir) / "sample_korean.md"
        normal_file.write_text("translated content", encoding='utf-8')
        
        # 정리 실행
        service.cleanup_temp_files(temp_dir)
        
        # 확인
        assert not raw_file.exists()  # 삭제됨
        assert not cache_file.exists()  # 삭제됨
        assert normal_file.exists()  # 보존됨

    @patch('services.translation_service.QualityContractTranslator')
    def test_singleton_translator_instance(self, mock_translator_class, service):
        """번역기 싱글톤 패턴 테스트"""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # 여러 번 호출해도 같은 인스턴스 반환
        translator1 = service._get_translator()
        translator2 = service._get_translator()
        
        assert translator1 is translator2
        assert mock_translator_class.call_count == 1  # 한 번만 생성됨


# 실행 예시:
# pytest tests/test_translation_service.py -v