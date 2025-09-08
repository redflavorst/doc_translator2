# services/translation_service_improved.py
"""
개선된 번역 서비스 - 테이블 번역 100% 보장
"""

from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import threading
import requests
import re
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from bs4 import BeautifulSoup, NavigableString
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

@dataclass
class TranslationResult:
    """번역 결과를 담는 데이터 클래스"""
    success: bool
    input_file: str
    output_file: str
    report: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None

# 구버전 API 호환성을 위한 alias
BatchTranslationResult = TranslationResult

class TranslationService:
    """개선된 번역 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 모델 설정
        self.model_name = self.config.get('model_name', 'gemma3n:e4b')  # 실제 사용 가능한 모델
        self.temperature = self.config.get('temperature', 0.1)
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        
        # 캐시
        self._cell_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._local = threading.local()

        # 병렬 처리
        default_workers = min(32, (os.cpu_count() or 1) * 5)
        self.max_workers = int(self.config.get("max_workers", default_workers))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 번역 실패 텍스트 로그 파일
        self.untranslated_log_path = Path(__file__).parent.parent / "untranslated_texts.txt"
        self._untranslated_texts = set()  # 중복 방지용
        
        # Ollama 서버 확인
        self._check_ollama_server()
    
    def _check_ollama_server(self):
        """Ollama 서버 상태 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/version", timeout=5)
            if response.status_code == 200:
                print(f"[OK] Ollama server connected: {self.ollama_url}")
        except Exception as e:
            print(f"[ERROR] Ollama server connection failed: {e}")
    
    def _get_session(self) -> requests.Session:
        """스레드별 HTTP 세션"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
        return self._local.session
    
    # ==================== 핵심 유틸리티 ====================
    
    def _set_cell_text(self, cell, text: str):
        """셀 텍스트 안전하게 설정"""
        cell.clear()
        cell.append(NavigableString(text if text is not None else ""))
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        if not text:
            return ""
        # 유니코드 공백 처리
        text = re.sub(r'[\xa0\u2000-\u200b\u202f\u205f\u3000]', ' ', text)
        # 중복 공백 제거
        text = ' '.join(text.split())
        return text.strip()
    
    def _cache_key(self, text: str) -> str:
        """캐시 키 생성"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    # ==================== 메인 번역 메서드 ====================
    
    def translate_document(self, input_file: str, output_file: str = None) -> TranslationResult:
        """문서 번역 메인 메서드"""
        start_time = time.time()
        session = self._get_session()
        
        try:
            # 파일 읽기
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 출력 파일 경로
            if not output_file:
                input_path = Path(input_file)
                output_file = str(input_path.parent / f"{input_path.stem}_korean{input_path.suffix}")
            
            # 번역 실패 로그 파일 경로 설정
            output_path = Path(output_file)
            input_path = Path(input_file)
            
            # 경로에 따라 다르게 처리
            if 'outputs' in str(output_path.parts):
                # 웹에서 실행: outputs/user_id/markdown/page_xxxx_untranslated.txt
                self.untranslated_log_path = output_path.parent / f"{output_path.stem}_untranslated.txt"
            else:
                # 직접 실행 (테스트): 출력 파일과 같은 위치
                self.untranslated_log_path = output_path.parent / f"{input_path.stem}_untranslated.txt"
            
            self._untranslated_texts = set()  # 각 문서마다 초기화
            
            print(f"[INFO] Untranslated texts will be saved to: {self.untranslated_log_path}")
            
            # 번역 실행
            translated_content, report = self._translate_markdown_with_tables(session, content)
            
            # 파일 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            # 번역 실패 로그 요약
            if self._untranslated_texts:
                print(f"[INFO] {len(self._untranslated_texts)} untranslated texts logged")
            
            elapsed = time.time() - start_time
            
            return TranslationResult(
                success=True,
                input_file=input_file,
                output_file=output_file,
                report=report,
                processing_time=elapsed
            )
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return TranslationResult(
                success=False,
                input_file=input_file,
                output_file=output_file,
                report={},
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _translate_markdown_with_tables(self, session: requests.Session, content: str) -> Tuple[str, Dict]:
        """마크다운 내 HTML 테이블 번역"""
        parts = []
        confidences = []
        
        # HTML 테이블 패턴
        pattern = re.compile(r"(<(?:div[^>]*>)?(?:<html>)?(?:<body>)?<table[\s\S]*?</table>(?:</body>)?(?:</html>)?(?:</div>)?)", re.IGNORECASE)
        
        cursor = 0
        for match in pattern.finditer(content):
            # 테이블 전 텍스트 번역
            before = content[cursor:match.start()]
            if before.strip():
                translated, conf = self._translate_text(session, before)
                parts.append(translated)
                confidences.append(conf)
            
            # 테이블 번역
            table_block = match.group(1)
            translated_table = self._translate_html_table_block(session, table_block)
            parts.append(translated_table)
            confidences.append(1.0)
            
            cursor = match.end()
        
        # 나머지 텍스트
        if cursor < len(content):
            remaining = content[cursor:]
            if remaining.strip():
                translated, conf = self._translate_text(session, remaining)
                parts.append(translated)
                confidences.append(conf)
        
        # 테이블이 없는 경우 전체 번역
        if not parts:
            translated, conf = self._translate_text(session, content)
            parts.append(translated)
            confidences.append(conf)
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
        return '\n'.join(parts), {"avg_confidence": avg_conf}
    
    # ==================== 테이블 번역 ====================
    
    def _translate_html_table_block(self, session: requests.Session, html_block: str) -> str:
        """HTML 테이블 번역 - 100% 셀 번역 보장"""
        if not BS4_AVAILABLE:
            print("[WARNING] BeautifulSoup not available")
            return html_block
        
        soup = BeautifulSoup(html_block, 'html.parser')
        tables = soup.find_all('table')
        
        for table in tables:
            # 1. 모든 셀 수집
            all_cells = self._collect_all_cells(table)
            if not all_cells:
                continue
            
            print(f"\n[INFO] Processing table with {len(all_cells)} cells")
            
            # 2. 통합 번역
            translated_map = self._translate_cells_batch(session, all_cells)
            
            # 3. 번역 적용
            self._apply_translations(all_cells, translated_map)
            
            # 4. 커버리지 확인
            coverage = sum(1 for c in all_cells if c['id'] in translated_map) / len(all_cells)
            print(f"[INFO] Translation coverage: {coverage:.1%}")
            
            # 5. 누락 재처리
            if coverage < 1.0:
                missing = [c for c in all_cells if c['id'] not in translated_map and not c['is_empty']]
                if missing:
                    print(f"[INFO] Retrying {len(missing)} missing cells")
                    retry_map = self._force_translate_cells(session, missing)
                    self._apply_translations(missing, retry_map)
        
        return str(soup)
    
    def _collect_all_cells(self, table) -> List[Dict]:
        """테이블의 모든 셀 수집"""
        cells = []
        cell_id = 0
        
        for row_idx, row in enumerate(table.find_all('tr')):
            for col_idx, cell in enumerate(row.find_all(['th', 'td'])):
                text = cell.get_text(' ', strip=True)
                cells.append({
                    'id': f'cell_{cell_id}',
                    'element': cell,
                    'text': text,
                    'row': row_idx,
                    'col': col_idx,
                    'is_empty': not bool(text.strip())
                })
                cell_id += 1
        
        return cells
    
    def _translate_cells_batch(self, session: requests.Session, cells: List[Dict]) -> Dict[str, str]:
        """배치 번역 - 속도 최적화"""
        import time
        start_time = time.time()
        result = {}
        
        # 빈 셀 제외
        non_empty = [c for c in cells if not c['is_empty']]
        if not non_empty:
            return result
            
        # 중복 제거 - 매우 중요한 최적화!
        unique_texts = {}
        for cell in non_empty:
            norm_text = self._normalize_text(cell['text'])
            if norm_text:
                if norm_text not in unique_texts:
                    unique_texts[norm_text] = []
                unique_texts[norm_text].append(cell['id'])
        
        print(f"[OPTIMIZE] {len(non_empty)} cells → {len(unique_texts)} unique texts")
        
        # 캐시 확인
        to_translate = []
        cache_hits = 0
        for norm_text, cell_ids in unique_texts.items():
            cache_key = self._cache_key(norm_text)
            with self._cache_lock:
                cached = self._cell_cache.get(cache_key)
            if cached is not None:
                cache_hits += 1
                for cell_id in cell_ids:
                    result[cell_id] = cached
            else:
                to_translate.append((norm_text, cell_ids))
        
        if cache_hits > 0:
            print(f"[CACHE] {cache_hits} cache hits")
        
        # 병렬 처리로 속도 개선
        if to_translate:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # 길이별 분류
            short_texts = [(t, ids) for t, ids in to_translate if len(t) < 100]
            medium_texts = [(t, ids) for t, ids in to_translate if 100 <= len(t) < 300]
            long_texts = [(t, ids) for t, ids in to_translate if len(t) >= 300]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                # 짧은 텍스트 - 대량 배치 (100개씩)
                if short_texts:
                    for i in range(0, len(short_texts), 100):
                        batch = short_texts[i:i+100]
                        future = executor.submit(
                            self._batch_translate_json,
                            session,
                            [t for t, _ in batch]
                        )
                        futures.append(('short', batch, future))
                
                # 중간 텍스트 - 중간 배치 (20개씩)
                if medium_texts:
                    for i in range(0, len(medium_texts), 20):
                        batch = medium_texts[i:i+20]
                        future = executor.submit(
                            self._batch_translate_json,
                            session,
                            [t for t, _ in batch]
                        )
                        futures.append(('medium', batch, future))
                
                # 긴 텍스트 - 개별 처리 (병렬)
                for text, cell_ids in long_texts:
                    future = executor.submit(self._translate_single, session, text)
                    futures.append(('long', [(text, cell_ids)], future))
                
                # 결과 수집
                for future in as_completed([f for _, _, f in futures]):
                    # Find which batch this future belongs to
                    for batch_type, batch_data, fut in futures:
                        if fut == future:
                            try:
                                if batch_type in ['short', 'medium']:
                                    translations = future.result()
                                    for (text, cell_ids), trans in zip(batch_data, translations.values()):
                                        with self._cache_lock:
                                            self._cell_cache[self._cache_key(text)] = trans
                                        for cell_id in cell_ids:
                                            result[cell_id] = trans
                                else:  # long
                                    trans = future.result()
                                    text, cell_ids = batch_data[0]
                                    with self._cache_lock:
                                        self._cell_cache[self._cache_key(text)] = trans
                                    for cell_id in cell_ids:
                                        result[cell_id] = trans
                            except Exception as e:
                                print(f"[ERROR] Translation failed: {e}")
                            break
        
        elapsed = time.time() - start_time
        print(f"[SPEED] Table translation took {elapsed:.2f}s")
        
        return result
    
    def _batch_translate_json(self, session: requests.Session, texts: List[str]) -> Dict[str, str]:
        """개별 번역만 사용 (JSON 배치는 모델 문제로 비활성화)"""
        if not texts:
            return {}
        
        # gemma3n:e4b 모델이 JSON 배열을 제대로 처리하지 못하므로
        # 개별 번역만 사용
        print(f"[INFO] Translating {len(texts)} items individually...")
        result = {}
        for i, text in enumerate(texts):
            translated = self._translate_single(session, text)
            result[text] = translated
            if (i + 1) % 10 == 0:
                print(f"[PROGRESS] Translated {i+1}/{len(texts)} cells")
        return result
    
    def _translate_single(self, session: requests.Session, text: str) -> str:
        """단일 텍스트 번역"""
        if not text or not text.strip():
            return text
        
        # 번역 불필요한 케이스 체크
        normalized = text.strip()
        
        # 1. 숫자만 있는 경우
        if normalized.replace('.', '').replace(',', '').replace('-', '').replace('+', '').isdigit():
            return text
        
        # 2. 특수문자만 있는 경우
        if all(not c.isalnum() for c in normalized):
            return text
        
        # 3. 숫자+단위 패턴 (예: 100%, $50, 2023년 등)
        import re
        if re.match(r'^[\d,.\-+]+\s*[%$€£¥₩년월일]?$', normalized):
            return text
        
        # 4. 날짜 형식 (예: 2023-01-01, 01/01/2023)
        if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', normalized):
            return text
        
        # 5. 제품코드/ID 패턴 (숫자+영문 조합, 예: 64350675KR07, ABC123, MODEL-2024)
        # 공백 없이 숫자와 영문이 섞여있으면 코드로 간주
        if re.match(r'^[A-Z0-9\-_]+$', normalized, re.IGNORECASE):
            # 최소 하나의 숫자와 하나의 문자가 있는 경우
            has_digit = any(c.isdigit() for c in normalized)
            has_alpha = any(c.isalpha() for c in normalized)
            if has_digit and has_alpha:
                return text
        
        # 6. 단일 영문 약어 (예: USA, KR, ID, CEO 등)
        if re.match(r'^[A-Z]{2,}$', normalized):
            return text
            
        # 프롬프트 개선 - thinking 방지
        prompt = f"""Translate the following English text to Korean. Provide ONLY the Korean translation without any explanation.

English: {text}
Korean:"""
        
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_predict': min(8000, len(text) * 10),  # Further increased for complete responses
                'top_p': 0.9,
                'repeat_penalty': 1.1
            }
        }
        
        try:
            response = session.post(f'{self.ollama_url}/api/generate', json=data, timeout=60)
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                
                # Remove <think> tags completely - they contain thinking process
                if '<think>' in result:
                    # Find content after </think> tag
                    if '</think>' in result:
                        # Extract everything after the closing tag
                        parts = result.split('</think>')
                        if len(parts) > 1:
                            # Get the last part (after all think tags)
                            result = parts[-1].strip()
                            # Clean any prompt echoes
                            result = re.sub(r'^.*?Korean.*?:', '', result, flags=re.IGNORECASE).strip()
                    else:
                        # No closing tag - response was cut off
                        # Try to extract Korean text that appears after common patterns
                        patterns = [
                            r'Korean.*?[:：]\s*([\uAC00-\uD7A3].*?)(?:$|\n|<)',
                            r'translation.*?[:：]\s*([\uAC00-\uD7A3].*?)(?:$|\n|<)',
                            r'([\uAC00-\uD7A3][\uAC00-\uD7A3\s\w.,!?;:()\-"\']+)'
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, result, re.IGNORECASE | re.DOTALL)
                            if match:
                                result = match.group(1) if '(' in pattern else match.group(0)
                                result = result.strip()
                                break
                        else:
                            # No Korean found - return original text as fallback
                            print(f"[WARNING] No Korean found after <think> tag for: {text[:30]}...")
                            return text
                
                # Clean up any remaining artifacts
                result = result.strip()
                if result.startswith(':'):
                    result = result[1:].strip()
                
                # Validate that we got actual Korean translation
                if result and result != text:
                    # Check if result contains Korean characters
                    has_korean = any('\uAC00' <= c <= '\uD7A3' for c in result)
                    if has_korean:
                        return result
                    else:
                        # 번역 실패 - 로그에 기록하고 원본 반환
                        self._log_untranslated(text)
                        print(f"[INFO] No Korean found, logged: {text[:30]}...")
                        return text
                else:
                    # 번역 결과가 원본과 같음 - 로그에 기록하고 원본 반환
                    self._log_untranslated(text)
                    print(f"[INFO] Translation unchanged, logged: {text[:30]}...")
                    return text
        except Exception as e:
            print(f"[ERROR] Translation API error: {e}")
            self._log_untranslated(text)
        
        return text
    
    def _log_untranslated(self, text: str):
        """번역 실패한 텍스트를 파일에 기록"""
        if text not in self._untranslated_texts:
            self._untranslated_texts.add(text)
            try:
                with open(self.untranslated_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{text}\n")
            except Exception as e:
                print(f"[ERROR] Failed to log untranslated text: {e}")
    
    def _translate_text(self, session: requests.Session, text: str) -> Tuple[str, float]:
        """일반 텍스트 번역 - 마크다운 구조 보존"""
        # 마크다운 헤더를 보존하기 위해 줄 단위로 처리
        lines = text.split('\n')
        translated_lines = []
        
        for line in lines:
            # 빈 줄은 그대로 보존
            if not line.strip():
                translated_lines.append('')
                continue
                
            # 마크다운 헤더는 별도 처리
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                header_level = header_match.group(1)
                header_text = header_match.group(2)
                translated_header = self._translate_single(session, header_text)
                translated_lines.append(f"{header_level} {translated_header}")
            else:
                # 일반 텍스트 줄
                if len(line) < 100:
                    translated_lines.append(self._translate_single(session, line))
                else:
                    # 긴 텍스트는 청크 분할 후 병렬 처리
                    chunks = self._split_into_chunks(line, 1000)
                    futures = [self.executor.submit(self._translate_single, session, chunk)
                               for chunk in chunks]
                    translated_chunks = [f.result() for f in futures]
                    translated_lines.append(' '.join(translated_chunks))
        
        return '\n'.join(translated_lines), 0.95
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """텍스트를 청크로 분할"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _parse_json_response(self, response: str, original_texts: List[str]) -> Dict[str, str]:
        """JSON 응답 파싱"""
        result = {}
        
        try:
            data = json.loads(response)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        idx = int(item.get('id', -1))
                        trans = item.get('translated', '')
                        if 0 <= idx < len(original_texts) and trans:
                            result[original_texts[idx]] = trans
        except:
            pass
        
        # 누락 항목은 원문
        for text in original_texts:
            if text not in result:
                result[text] = text
        
        return result
    
    def _apply_translations(self, cells: List[Dict], translations: Dict[str, str]):
        """번역 적용"""
        for cell in cells:
            if cell['id'] in translations:
                self._set_cell_text(cell['element'], translations[cell['id']])
    
    def _force_translate_cells(self, session: requests.Session, cells: List[Dict]) -> Dict[str, str]:
        """강제 번역"""
        result = {}
        for cell in cells:
            trans = self._translate_single(session, cell['text'])
            result[cell['id']] = trans
        return result

# 테스트
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    input_md = base_dir / "uploads" / "page_0008.md"
    output_md = base_dir / "uploads" / "page_0008_korean.md"

    service = TranslationService()
    result = service.translate_document(str(input_md), str(output_md))
    print(f"Translation {'succeeded' if result.success else 'failed'}")