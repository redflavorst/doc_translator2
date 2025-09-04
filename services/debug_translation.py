"""
Translation Service 디버깅 스크립트
특정 지점에서 데이터를 확인하기 위한 테스트
"""

import sys
import json
from pathlib import Path
from translation_service import TranslationService
from bs4 import BeautifulSoup, NavigableString
import pdb  # Python debugger

class DebugTranslationService(TranslationService):
    """디버깅용 TranslationService - 메소드 오버라이드"""
    
    def _translate_html_table_block(self, session, html_block: str) -> str:
        """HTML 테이블 번역 - 디버깅 버전"""
        print("\n" + "="*60)
        print("[DEBUG] Entering _translate_html_table_block")
        print("="*60)
        
        # HTML 블록 내용 확인
        print(f"\n[DEBUG] HTML block length: {len(html_block)} characters")
        print(f"[DEBUG] First 500 chars of HTML block:\n{html_block[:500]}")
        
        if not self.BS4_AVAILABLE:
            print("[WARNING] BeautifulSoup not available")
            return html_block
        
        # HTML 파싱
        soup = BeautifulSoup(html_block, 'html.parser')
        tables = soup.find_all('table')
        print(f"\n[DEBUG] Found {len(tables)} table(s)")
        
        for table_idx, table in enumerate(tables):
            print(f"\n[DEBUG] Processing table #{table_idx + 1}")
            
            # 1. 모든 셀 수집
            all_cells = self._collect_all_cells(table)
            
            # 셀 데이터 상세 출력
            print(f"\n[DEBUG] Collected {len(all_cells)} cells")
            print("\n[DEBUG] Cell details:")
            for i, cell in enumerate(all_cells[:10]):  # 처음 10개만 표시
                print(f"  Cell {i}: ID={cell['id']}, Row={cell['row']}, Col={cell['col']}")
                print(f"    Text: '{cell['text'][:50]}...' (length: {len(cell['text'])})")
                print(f"    Empty: {cell['is_empty']}")
            
            if len(all_cells) > 10:
                print(f"  ... and {len(all_cells) - 10} more cells")
            
            # 중단점 설정 옵션
            user_input = input("\n[DEBUG] Press Enter to continue to _translate_cells_batch, or 'pdb' to enter debugger: ")
            if user_input.lower() == 'pdb':
                pdb.set_trace()  # 여기서 Python 디버거 시작
            
            # 2. 통합 번역
            print("\n[DEBUG] Calling _translate_cells_batch...")
            translated_map = self._translate_cells_batch(session, all_cells)
            
            print(f"\n[DEBUG] Translation map contains {len(translated_map)} entries")
            for key, value in list(translated_map.items())[:5]:  # 처음 5개만 표시
                print(f"  {key}: '{value[:50]}...'")
            
            # 나머지는 원본 메소드 로직 계속...
            self._apply_translations(all_cells, translated_map)
            
            coverage = sum(1 for c in all_cells if c['id'] in translated_map) / len(all_cells)
            print(f"\n[DEBUG] Translation coverage: {coverage:.1%}")
            
            if coverage < 1.0:
                missing = [c for c in all_cells if c['id'] not in translated_map and not c['is_empty']]
                if missing:
                    print(f"[DEBUG] {len(missing)} cells missing translation")
                    retry_map = self._force_translate_cells(session, missing)
                    self._apply_translations(missing, retry_map)
        
        return str(soup)

def test_with_sample():
    """샘플 HTML 테이블로 테스트"""
    
    # 테스트용 마크다운 파일 생성
    test_content = """# Test Document

This is a test paragraph.

<table>
<tr>
<th>Name</th>
<th>Age</th>
<th>City</th>
</tr>
<tr>
<td>John</td>
<td>30</td>
<td>New York</td>
</tr>
<tr>
<td>Jane</td>
<td>25</td>
<td>London</td>
</tr>
</table>

Another paragraph here.
"""
    
    # 테스트 파일 저장
    test_file = Path("test_debug.md")
    test_file.write_text(test_content, encoding='utf-8')
    
    # 디버깅용 서비스 생성
    service = DebugTranslationService()
    
    print("[TEST] Starting translation test...")
    result = service.translate_document(
        str(test_file),
        "test_debug_output.md"
    )
    
    if result.success:
        print(f"\n[TEST] Translation succeeded!")
        print(f"[TEST] Output file: {result.output_file}")
        print(f"[TEST] Processing time: {result.processing_time:.2f}s")
    else:
        print(f"\n[TEST] Translation failed: {result.error}")

def test_with_existing_file():
    """기존 파일로 테스트"""
    
    # 디버깅용 서비스 생성
    service = DebugTranslationService()
    
    # 실제 파일 경로 사용
    input_file = "../uploads/page_0003.md"
    output_file = "../uploads/page_0003_debug.md"
    
    print(f"[TEST] Testing with {input_file}")
    result = service.translate_document(input_file, output_file)
    
    if result.success:
        print(f"\n[TEST] Translation succeeded!")
        print(f"[TEST] Output file: {result.output_file}")
    else:
        print(f"\n[TEST] Translation failed: {result.error}")

if __name__ == "__main__":
    print("Select test mode:")
    print("1. Test with sample HTML table")
    print("2. Test with existing file (../uploads/page_0003.md)")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == "1":
        test_with_sample()
    elif choice == "2":
        test_with_existing_file()
    else:
        print("Invalid choice")