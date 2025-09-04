# services/pdf_converter_service.py
"""
마크다운 파일들을 하나의 PDF로 변환하는 서비스
wkhtmltopdf를 사용하여 각 page_xxxx_korean.md 파일을 개별 PDF로 변환 후 병합
"""

import os
import re
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import fitz  # PyMuPDF for PDF manipulation
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False


@dataclass
class PDFConversionResult:
    """PDF 변환 결과를 담는 데이터 클래스"""
    success: bool
    output_file: str
    total_pages: int
    processing_time: float
    error: Optional[str] = None
    warnings: List[str] = None


class PDFConverterService:
    """마크다운 파일을 PDF로 변환하는 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # wkhtmltopdf 설정 - Windows의 경우 전체 경로 지정
        import sys
        if sys.platform == 'win32':
            # Windows에서 wkhtmltopdf 일반 설치 경로들
            possible_paths = [
                r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
                r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
                'wkhtmltopdf'  # PATH에 있는 경우
            ]
            
            # 실제 존재하는 경로 찾기
            import os
            wkhtmltopdf_found = None
            for path in possible_paths:
                expanded_path = os.path.expandvars(path)
                if os.path.exists(expanded_path) or path == 'wkhtmltopdf':
                    wkhtmltopdf_found = path
                    break
            
            self.wkhtmltopdf_path = self.config.get('wkhtmltopdf_path', wkhtmltopdf_found or 'wkhtmltopdf')
        else:
            self.wkhtmltopdf_path = self.config.get('wkhtmltopdf_path', 'wkhtmltopdf')
        
        # 한글 폰트 설정 (Windows/Mac/Linux별 기본 폰트)
        self.default_fonts = {
            'win32': 'Malgun Gothic',  # 맑은 고딕
            'darwin': 'Apple SD Gothic Neo',  # Mac 한글 폰트
            'linux': 'Noto Sans CJK KR'  # Linux 한글 폰트
        }
        
        # 사용자 지정 폰트 또는 OS별 기본 폰트 사용
        import sys
        platform = sys.platform
        self.main_font = self.config.get('main_font', 
                                         self.default_fonts.get(platform, 'Noto Sans CJK KR'))
        
        # wkhtmltopdf 설치 확인
        self._check_wkhtmltopdf()
    
    def _check_wkhtmltopdf(self):
        """wkhtmltopdf 설치 여부 확인"""
        try:
            result = subprocess.run(
                [self.wkhtmltopdf_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.split('\n')[0] if result.stdout else "Unknown"
                self.logger.info(f"wkhtmltopdf found: {version}")
                print(f"[OK] wkhtmltopdf 설치 확인: {version}")
        except Exception as e:
            self.logger.warning(f"wkhtmltopdf 확인 실패: {e}")
            print("[WARNING] wkhtmltopdf를 찾을 수 없습니다.")
            print("   설치: https://wkhtmltopdf.org/downloads.html")
    
    def _basic_markdown_to_html(self, markdown_content: str) -> str:
        """기본적인 마크다운을 HTML로 변환 (라이브러리 없이)"""
        import re
        
        html = markdown_content
        
        # 헤더 변환 (## → <h2>)
        html = re.sub(r'^#{6}\s+(.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
        html = re.sub(r'^#{5}\s+(.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
        html = re.sub(r'^#{4}\s+(.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^#{3}\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^#{2}\s+(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^#{1}\s+(.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # 굵은 글씨 (**text** 또는 __text__)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
        
        # 기울임체 (*text* 또는 _text_)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
        
        # 코드 블록
        html = re.sub(r'```(.+?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
        
        # 테이블 변환
        table_lines = []
        in_table = False
        lines_for_table = html.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines_for_table):
            # 테이블 구분선 검사 (|---|---|)
            if re.match(r'^\s*\|[\s\-:|]+\|', line):
                if i > 0 and '|' in lines_for_table[i-1]:
                    # 이전 줄이 헤더
                    if not in_table:
                        processed_lines.pop()  # 헤더 줄 제거 (다시 처리하기 위해)
                        header_line = lines_for_table[i-1]
                        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
                        processed_lines.append('<table>')
                        processed_lines.append('<thead><tr>')
                        for header in headers:
                            processed_lines.append(f'<th>{header}</th>')
                        processed_lines.append('</tr></thead>')
                        processed_lines.append('<tbody>')
                        in_table = True
                continue
            elif '|' in line and in_table:
                # 테이블 데이터 행
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                processed_lines.append('<tr>')
                for cell in cells:
                    processed_lines.append(f'<td>{cell}</td>')
                processed_lines.append('</tr>')
            elif in_table and '|' not in line:
                # 테이블 종료
                processed_lines.append('</tbody>')
                processed_lines.append('</table>')
                in_table = False
                processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        # 마지막에 테이블이 열려있으면 닫기
        if in_table:
            processed_lines.append('</tbody>')
            processed_lines.append('</table>')
        
        html = '\n'.join(processed_lines)
        
        # 리스트 처리
        lines = html.split('\n')
        in_ul = False
        in_ol = False
        new_lines = []
        
        for line in lines:
            # 순서 없는 리스트
            if re.match(r'^\s*[-*+]\s+', line):
                if not in_ul:
                    new_lines.append('<ul>')
                    in_ul = True
                content = re.sub(r'^\s*[-*+]\s+', '', line)
                new_lines.append(f'<li>{content}</li>')
            # 순서 있는 리스트
            elif re.match(r'^\s*\d+\.\s+', line):
                if not in_ol:
                    new_lines.append('<ol>')
                    in_ol = True
                content = re.sub(r'^\s*\d+\.\s+', '', line)
                new_lines.append(f'<li>{content}</li>')
            else:
                if in_ul:
                    new_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    new_lines.append('</ol>')
                    in_ol = False
                new_lines.append(line)
        
        # 마지막에 열린 리스트 태그 닫기
        if in_ul:
            new_lines.append('</ul>')
        if in_ol:
            new_lines.append('</ol>')
        
        html = '\n'.join(new_lines)
        
        # 단락 처리 (빈 줄로 구분된 텍스트를 <p>로 감싸기)
        paragraphs = html.split('\n\n')
        new_paragraphs = []
        for para in paragraphs:
            # HTML 태그가 없는 일반 텍스트면 <p>로 감싸기
            if para.strip() and not re.match(r'^\s*<', para.strip()):
                new_paragraphs.append(f'<p>{para.strip()}</p>')
            else:
                new_paragraphs.append(para)
        
        html = '\n'.join(new_paragraphs)
        
        # 줄바꿈을 <br>로 변환 (단일 줄바꿈)
        html = re.sub(r'(?<!\n)\n(?!\n)', '<br>\n', html)
        
        return html
    
    def find_korean_markdown_files(self, directory: str) -> List[Path]:
        """
        디렉토리에서 page_xxxx_korean.md 파일들을 찾아 정렬하여 반환
        하위 디렉토리까지 재귀적으로 검색
        
        Args:
            directory: 검색할 디렉토리 경로
            
        Returns:
            정렬된 마크다운 파일 경로 리스트
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")
        
        # page_xxxx_korean.md 패턴 매칭 (하위 디렉토리 포함)
        pattern = re.compile(r'page_(\d+)_korean\.md$')
        korean_files = []
        
        # 재귀적으로 모든 하위 디렉토리 검색
        for file in directory_path.rglob('*_korean.md'):
            match = pattern.search(file.name)
            if match:
                page_num = int(match.group(1))
                korean_files.append((page_num, file))
        
        # 페이지 번호순 정렬
        korean_files.sort(key=lambda x: x[0])
        sorted_files = [file for _, file in korean_files]
        
        self.logger.info(f"{len(sorted_files)}개의 한글 마크다운 파일을 찾았습니다")
        for file in sorted_files[:3]:  # 처음 3개만 로그로 출력
            self.logger.info(f"  - {file}")
        return sorted_files

    def markdown_to_html(self, markdown_content: str, title: str = "Document") -> str:
        """마크다운을 HTML로 변환 (스타일 포함)"""
        try:
            import markdown
            # 확장 기능을 추가하여 더 많은 마크다운 문법 지원
            md = markdown.Markdown(extensions=[
                'tables',           # 테이블 지원
                'fenced_code',      # 코드 블록 지원
                'nl2br',            # 줄바꿈을 <br>로 변환
                'extra',            # 추가 마크다운 기능
                'codehilite',       # 코드 하이라이팅
                'toc',              # 목차 지원
                'sane_lists',       # 더 나은 리스트 처리
                'attr_list',        # 속성 추가 지원
                'md_in_html',       # HTML 내 마크다운
            ])
            
            # 마크다운 변환
            html_body = md.convert(markdown_content)
            
            # 디버깅용 로그
            self.logger.info(f"마크다운 변환 완료. 원본 첫 100자: {markdown_content[:100]}")
            self.logger.info(f"HTML 변환 결과 첫 200자: {html_body[:200]}")
            
        except ImportError as e:
            self.logger.warning(f"markdown 라이브러리 import 실패: {e}")
            # markdown 라이브러리가 없으면 기본 변환
            html_body = self._basic_markdown_to_html(markdown_content)
            
        except Exception as e:
            self.logger.error(f"마크다운 변환 중 오류: {e}")
            # 오류 발생 시 기본 변환 사용
            html_body = self._basic_markdown_to_html(markdown_content)
        
        # HTML 템플릿 (한글 폰트 및 스타일 포함)
        html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: '{self.main_font}', '맑은 고딕', 'Nanum Gothic', sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            margin: 20px;
            padding: 0;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 0.5em;
            margin-bottom: 0.3em;
        }}
        h1 {{ font-size: 22pt; font-weight: 700; }}
        h2 {{ font-size: 18pt; font-weight: 700; }}
        h3 {{ font-size: 16pt; font-weight: 700; }}
        h4, h5, h6 {{ font-size: 14pt; font-weight: 700; }}  /* 작은 헤더 최소 크기 보장 */
        p {{
            margin: 0.5em 0;
            text-align: justify;
            word-break: keep-all;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 10pt;
        }}
        table td, table th {{
            border: 1px solid #999;
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
        }}
        table th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        ul, ol {{
            margin: 0.5em 0 0.5em 1.5em;
            padding: 0;
        }}
        li {{
            margin: 0.3em 0;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            font-family: Consolas, monospace;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""
        return html_template

    def convert_markdown_to_individual_pdf(self, markdown_file: Path, output_pdf: Path) -> bool:
        """단일 마크다운 파일을 PDF로 변환 (wkhtmltopdf 사용)"""
        try:
            # 마크다운 파일 읽기
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # HTML로 변환
            html_content = self.markdown_to_html(markdown_content, title=markdown_file.stem)
            
            # 임시 HTML 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.html', delete=False) as temp_html:
                temp_html.write(html_content)
                temp_html_path = temp_html.name
            
            try:
                # wkhtmltopdf 명령어 구성
                wkhtmltopdf_cmd = [
                    self.wkhtmltopdf_path,
                    '--encoding', 'utf-8',
                    '--page-size', 'A4',
                    '--margin-top', '15mm',
                    '--margin-bottom', '15mm',
                    '--margin-left', '15mm',
                    '--margin-right', '15mm',
                    '--minimum-font-size', '10',
                    '--enable-local-file-access',
                    '--print-media-type',
                    '--no-header-line',
                    '--no-footer-line',
                    temp_html_path,
                    str(output_pdf)
                ]
                
                # PDF 생성 실행
                result = subprocess.run(
                    wkhtmltopdf_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return True
                else:
                    self.logger.error(f"wkhtmltopdf 에러: {result.stderr}")
                    return False
                    
            finally:
                # 임시 HTML 파일 삭제
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)
                    
        except Exception as e:
            self.logger.error(f"PDF 변환 실패 ({markdown_file}): {e}")
            return False

    def merge_pdfs(self, pdf_files: List[Path], output_pdf: Path) -> bool:
        """여러 PDF 파일을 하나로 병합 (PyMuPDF 사용)"""
        if not PYMUPDF_AVAILABLE:
            self.logger.error("PyMuPDF가 설치되지 않았습니다. pip install PyMuPDF")
            return False
        
        try:
            # 새 PDF 문서 생성
            merged_pdf = fitz.open()
            
            # 각 PDF 파일 병합
            for pdf_file in pdf_files:
                if pdf_file.exists():
                    pdf = fitz.open(str(pdf_file))
                    merged_pdf.insert_pdf(pdf)
                    pdf.close()
            
            # 병합된 PDF 저장
            merged_pdf.save(str(output_pdf))
            merged_pdf.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"PDF 병합 실패: {e}")
            return False

    def convert_to_pdf(self, 
                       input_directory: str,
                       output_pdf_path: str = None,
                       include_metadata: bool = True,
                       add_page_breaks: bool = True,
                       pdf_metadata: Optional[Dict[str, Any]] = None) -> PDFConversionResult:
        """
        각 마크다운 파일을 개별 PDF로 변환 후 병합
        1. page_xxxx_korean.md → page_xxxx_korean.pdf (각각 변환)
        2. 모든 PDF 파일 병합 → translated_document.pdf
        
        Args:
            input_directory: 마크다운 파일들이 있는 디렉토리
            output_pdf_path: 출력 PDF 파일 경로 (None이면 자동 생성)
            include_metadata: 메타데이터 포함 여부
            add_page_breaks: 페이지 구분 추가 여부
            pdf_metadata: 원본 PDF 메타데이터
            
        Returns:
            PDFConversionResult: 변환 결과
        """
        import time
        start_time = time.time()
        
        try:
            # 1. 한글 마크다운 파일 찾기
            markdown_files = self.find_korean_markdown_files(input_directory)
            if not markdown_files:
                raise ValueError(f"한글 마크다운 파일을 찾을 수 없습니다: {input_directory}")
            
            print(f"[INFO] {len(markdown_files)}개의 마크다운 파일을 개별 PDF로 변환합니다...")
            
            # 2. 출력 파일 경로 설정
            if output_pdf_path is None:
                if 'markdown' in str(input_directory):
                    output_base_dir = Path(input_directory).parent
                else:
                    output_base_dir = Path(input_directory)
                
                output_pdf_path = os.path.join(
                    str(output_base_dir), 
                    f"translated_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
            
            # 3. 임시 PDF 파일들을 저장할 디렉토리 생성
            temp_pdf_dir = Path(tempfile.mkdtemp(prefix="pdf_conversion_"))
            individual_pdfs = []
            warnings = []
            
            try:
                # 4. 각 마크다운 파일을 개별 PDF로 변환
                for i, md_file in enumerate(markdown_files, 1):
                    print(f"  [{i}/{len(markdown_files)}] {md_file.name} -> PDF 변환 중...")
                    
                    # 개별 PDF 파일 경로
                    temp_pdf = temp_pdf_dir / f"page_{i:04d}.pdf"
                    
                    # 마크다운 → PDF 변환
                    if self.convert_markdown_to_individual_pdf(md_file, temp_pdf):
                        individual_pdfs.append(temp_pdf)
                        print(f"    [OK] 변환 완료: {temp_pdf.name}")
                    else:
                        warnings.append(f"변환 실패: {md_file.name}")
                        print(f"    [FAIL] 변환 실패: {md_file.name}")
                
                if not individual_pdfs:
                    raise RuntimeError("PDF 변환에 모두 실패했습니다.")
                
                print(f"\n[INFO] {len(individual_pdfs)}개의 PDF를 하나로 병합 중...")
                
                # 5. 모든 개별 PDF를 하나로 병합
                if self.merge_pdfs(individual_pdfs, Path(output_pdf_path)):
                    print(f"[OK] PDF 병합 완료: {output_pdf_path}")
                else:
                    raise RuntimeError("PDF 병합 실패")
                
            finally:
                # 6. 임시 파일들 정리
                import shutil
                if temp_pdf_dir.exists():
                    shutil.rmtree(temp_pdf_dir, ignore_errors=True)
                    print("[CLEANUP] 임시 파일 정리 완료")
            
            # 7. 성공 결과 반환
            processing_time = time.time() - start_time
            
            # PDF 파일 크기 확인
            pdf_size = os.path.getsize(output_pdf_path) if os.path.exists(output_pdf_path) else 0
            total_pages = len(individual_pdfs)
            
            print(f"\n[SUCCESS] PDF 변환 완료!")
            print(f"   출력 파일: {output_pdf_path}")
            print(f"   총 페이지: {total_pages}")
            print(f"   파일 크기: {pdf_size / (1024*1024):.2f} MB")
            print(f"   처리 시간: {processing_time:.2f}초")
            
            return PDFConversionResult(
                success=True,
                output_file=str(output_pdf_path),
                total_pages=total_pages,
                processing_time=processing_time,
                warnings=warnings if warnings else None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"PDF 변환 실패: {e}")
            
            return PDFConversionResult(
                success=False,
                output_file="",
                total_pages=0,
                processing_time=processing_time,
                error=str(e)
            )


# 테스트 코드 (직접 실행 시)
if __name__ == "__main__":
    import sys
    
    # 테스트용 디렉토리 경로
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = input("번역된 마크다운 파일들이 있는 디렉토리 경로를 입력하세요: ")
    
    # PDF 변환 서비스 초기화
    converter = PDFConverterService()
    
    # PDF 변환 실행
    result = converter.convert_to_pdf(
        input_dir,
        include_metadata=True,
        add_page_breaks=True
    )
    
    if result.success:
        print(f"[SUCCESS] PDF 생성 성공: {result.output_file}")
        print(f"   총 {result.total_pages}페이지")
        print(f"   처리 시간: {result.processing_time:.2f}초")
    else:
        print(f"[FAIL] PDF 생성 실패: {result.error}")