import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Tuple

import requests

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

# Hard-coded Ollama configuration
OLLAMA_URL = 'http://localhost:11434'
MODEL_NAME = 'gemma3n:e4b'

# Ensure project root is on sys.path so that 'services' package is importable
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Regex used in services.translation_service._translate_markdown_with_tables
TABLE_PATTERN_STR = r"(<(?:div[^>]*>)?(?:<html>)?(?:<body>)?<table[\s\S]*?</table>(?:</body>)?(?:</html>)?(?:</div>)?)"

# Default markdown directory (falls back here when no CLI path is provided)
DEFAULT_MD_DIR = r'D:\PythonProject\doc_translator2\outputs\user_1_20250829_135304\markdown\page_0003_0'


def extract_table_blocks(content: str) -> List[Tuple[int, int, str]]:
    pattern = re.compile(TABLE_PATTERN_STR, re.IGNORECASE)
    blocks = []
    for m in pattern.finditer(content):
        blocks.append((m.start(), m.end(), m.group(1)))
    return blocks


def non_table_segments(content: str, blocks: List[Tuple[int, int, str]]) -> List[str]:
    segs = []
    cursor = 0
    for start, end, _ in blocks:
        if cursor < start:
            segs.append(content[cursor:start])
        cursor = end
    if cursor < len(content):
        segs.append(content[cursor:])
    return segs


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\xa0\u2000-\u200b\u202f\u205f\u3000]", " ", text)
    text = " ".join(text.split())
    return text.strip()


def analyze_tables(html_blocks: List[str]):
    results = []
    if not BS4_AVAILABLE:
        return results
    for html in html_blocks:
        soup = BeautifulSoup(html, 'html.parser')
        cells = []
        for row in soup.find_all('tr'):
            for cell in row.find_all(['th', 'td']):
                txt = cell.get_text(' ', strip=True)
                cells.append(txt)
        non_empty = [c for c in cells if c.strip()]
        unique = {}
        for c in non_empty:
            n = normalize_text(c)
            if not n:
                continue
            unique.setdefault(n, 0)
            unique[n] += 1
        results.append({
            'total_cells': len(cells),
            'non_empty_cells': len(non_empty),
            'unique_texts': len(unique),
            'top_examples': list(unique.keys())[:10]
        })
    return results


def collect_cells_with_coords(html: str):
    """서비스 로직과 동일한 순서로 셀을 수집(id, row, col, text, is_empty)."""
    if not BS4_AVAILABLE:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    cells = []
    cell_id = 0
    for row_idx, row in enumerate(soup.find_all('tr')):
        for col_idx, cell in enumerate(row.find_all(['th', 'td'])):
            text = cell.get_text(' ', strip=True)
            cells.append({
                'id': f'cell_{cell_id}',
                'row': row_idx,
                'col': col_idx,
                'text': text,
                'is_empty': not bool(text.strip())
            })
            cell_id += 1
    return cells


def dump_cells_markdown(cells: List[dict], out_path: Path, title: str):
    """셀 목록을 마크다운 테이블로 저장."""
    lines = [f"# {title}", "", "| id | row | col | is_empty | text |", "|---|---:|---:|:---:|---|"]
    for c in cells:
        text = c.get('text', '')
        # 파이프/개행 이스케이프
        safe_text = text.replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {c['id']} | {c['row']} | {c['col']} | {str(c['is_empty']).lower()} | {safe_text} |")
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def split_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = []
    size = 0
    for s in sentences:
        if size + len(s) > chunk_size and current:
            chunks.append(' '.join(current))
            current = [s]
            size = len(s)
        else:
            current.append(s)
            size += len(s)
    if current:
        chunks.append(' '.join(current))
    return chunks if chunks else [text]


def analyze_non_table_segments(segments: List[str]):
    analyses = []
    header_re = re.compile(r'^(#{1,6})\s+(.+)$')
    for seg in segments:
        lines = seg.split('\n')
        calls = 0
        details = []
        for line in lines:
            if not line.strip():
                continue
            m = header_re.match(line)
            if m:
                calls += 1
                details.append({'type': 'header', 'text_len': len(m.group(2))})
            else:
                if len(line) < 100:
                    calls += 1
                    details.append({'type': 'line', 'text_len': len(line)})
                else:
                    chks = split_into_chunks(line, 1000)
                    calls += len(chks)
                    details.append({'type': 'long_line', 'chunks': len(chks), 'text_len': len(line)})
        analyses.append({'estimated_llm_calls': calls, 'lines_analyzed': len(lines), 'details_sample': details[:10]})
    return analyses


class LoggingSession(requests.Session):
    def __init__(self, log_dir: Path, dry_run: bool = False):
        super().__init__()
        self.log_dir = log_dir
        self.counter = 0
        self.dry_run = dry_run

    def post(self, url, *args, **kwargs):  # type: ignore[override]
        self.counter += 1
        req_json = kwargs.get('json')
        (self.log_dir / f"llm_request_{self.counter:04d}.json").write_text(
            json.dumps({'url': url, 'json': req_json}, ensure_ascii=False, indent=2), encoding='utf-8')
        if self.dry_run:
            resp = requests.Response()
            resp.status_code = 200
            preview = ''
            try:
                prompt = (req_json or {}).get('prompt') or ''
                preview = prompt[:120]
            except Exception:
                pass
            resp._content = json.dumps({'response': f'[DRY-RUN]{preview}...'}).encode('utf-8')
            (self.log_dir / f"llm_response_{self.counter:04d}.json").write_text(
                json.dumps({'status': 200, 'body': json.loads(resp.content.decode('utf-8'))}, ensure_ascii=False, indent=2), encoding='utf-8')
            # Save raw text as-is and print to console
            try:
                raw_text = resp.text
            except Exception:
                raw_text = ''
            (self.log_dir / f"llm_response_{self.counter:04d}.txt").write_text(raw_text, encoding='utf-8')
            print(f"\n===== LLM RESPONSE {self.counter:04d} (RAW) =====\n{raw_text}\n===== END RESPONSE {self.counter:04d} =====\n")
            return resp
        start = time.time()
        resp = super().post(url, *args, **kwargs)
        elapsed = time.time() - start
        try:
            body = resp.text
        except Exception:
            body = ''
        (self.log_dir / f"llm_response_{self.counter:04d}.json").write_text(
            json.dumps({'status': resp.status_code, 'elapsed_sec': elapsed, 'body': body}, ensure_ascii=False, indent=2), encoding='utf-8')
        # Save raw text as-is and print to console
        (self.log_dir / f"llm_response_{self.counter:04d}.txt").write_text(body, encoding='utf-8')
        print(f"\n===== LLM RESPONSE {self.counter:04d} (RAW) =====\n{body}\n===== END RESPONSE {self.counter:04d} =====\n")
        return resp


def main():
    parser = argparse.ArgumentParser(description='Debug Markdown translation flow (preprocess + LLM calls + outputs).')
    parser.add_argument('markdown_file', nargs='?', default=None, help='Input markdown file path (optional)')
    parser.add_argument('--dry-run', action='store_true', help='Do not call real LLM; return stub responses')
    parser.add_argument('--out-dir', default=None, help='Debug output directory (default: outputs/debug_flow/<timestamp>)')
    parser.add_argument('--pattern-only', action='store_true', help='Stop after regex match; dump only matched table blocks and summary')
    args = parser.parse_args()

    if args.markdown_file:
        md_path = Path(args.markdown_file).resolve()
    else:
        # Auto-pick first .md in DEFAULT_MD_DIR
        default_dir = Path(DEFAULT_MD_DIR)
        if not default_dir.exists():
            raise FileNotFoundError(f"DEFAULT_MD_DIR not found: {DEFAULT_MD_DIR}")
        md_files = list(default_dir.glob('*.md'))
        if not md_files:
            raise FileNotFoundError(f"No .md files in DEFAULT_MD_DIR: {DEFAULT_MD_DIR}")
        md_path = md_files[0].resolve()
        print(f"[INFO] Using default markdown file: {md_path}")
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    timestamp = int(time.time())
    debug_dir = Path(args.out_dir) if args.out_dir else Path('outputs') / 'debug_flow' / f'{timestamp}'
    debug_dir.mkdir(parents=True, exist_ok=True)

    content = md_path.read_text(encoding='utf-8')

    # 1) Preprocess inspection
    blocks = extract_table_blocks(content)
    html_list = [b[2] for b in blocks]
    segs = non_table_segments(content, blocks)
    table_stats = analyze_tables(html_list)
    seg_stats = analyze_non_table_segments(segs)

    # Save raw inputs for reference
    (debug_dir / 'input_markdown.md').write_text(content, encoding='utf-8')
    for i, html in enumerate(html_list, 1):
        (debug_dir / f'table_{i:04d}.html').write_text(html, encoding='utf-8')
        # Dump collected cells (pre-translation)
        cells = collect_cells_with_coords(html)
        dump_cells_markdown(cells, debug_dir / f'table_{i:04d}_cells.md', f'Table {i} cells (pre-translation)')
    (debug_dir / 'preprocess_summary.json').write_text(
        json.dumps({'tables_found': len(html_list), 'table_stats': table_stats, 'non_table_segments': len(segs), 'non_table_stats': seg_stats}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    # Save pattern string for verification
    (debug_dir / 'pattern.txt').write_text(TABLE_PATTERN_STR, encoding='utf-8')

    # If only checking regex pattern/matches, stop here
    if args.pattern_only:
        summary = {
            'mode': 'pattern-only',
            'input_file': str(md_path),
            'tables_found': len(html_list),
            'output_dir': str(debug_dir)
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # 2) Wire service with logging session
    from services.translation_service import TranslationService

    service = TranslationService({
        'model_name': MODEL_NAME,
        'ollama_url': OLLAMA_URL,
        'temperature': 0.1
    })

    # Inject logging session (captures requests to Ollama)
    logging_session = LoggingSession(debug_dir, dry_run=args.dry_run)
    setattr(service._local, 'session', logging_session)

    # 3) Run translation
    output_md = debug_dir / f"{md_path.stem}_korean{md_path.suffix}"
    result = service.translate_document(str(md_path), str(output_md))

    # 4) Post-check: capture translated table blocks for comparison
    translated_content = output_md.read_text(encoding='utf-8') if output_md.exists() else ''
    translated_blocks = extract_table_blocks(translated_content)
    for i, (_, _, html) in enumerate(translated_blocks, 1):
        (debug_dir / f'table_{i:04d}_translated.html').write_text(html, encoding='utf-8')
        # Dump collected cells (post-translation)
        cells_tr = collect_cells_with_coords(html)
        dump_cells_markdown(cells_tr, debug_dir / f'table_{i:04d}_translated_cells.md', f'Table {i} cells (post-translation)')

    # Final summary
    summary = {
        'input_file': str(md_path),
        'output_file': str(output_md),
        'tables_found': len(html_list),
        'translated_tables_found': len(translated_blocks),
        'llm_calls_logged': logging_session.counter,
        'service_report': result.report if hasattr(result, 'report') else {}
    }
    (debug_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


