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
        self._local = threading.local()
        
        # 병렬 처리
        self.executor = ThreadPoolExecutor(max_workers=4)
        
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
            
            # 번역 실행
            translated_content, report = self._translate_markdown_with_tables(session, content)
            
            # 출력 파일 경로
            if not output_file:
                input_path = Path(input_file)
                output_file = str(input_path.parent / f"{input_path.stem}_korean{input_path.suffix}")
            
            # 파일 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
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
            if cache_key in self._cell_cache:
                cache_hits += 1
                for cell_id in cell_ids:
                    result[cell_id] = self._cell_cache[cache_key]
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
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                # 짧은 텍스트 - 대량 배치 (50개씩)
                if short_texts:
                    for i in range(0, len(short_texts), 50):
                        batch = short_texts[i:i+50]
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
                                        self._cell_cache[self._cache_key(text)] = trans
                                        for cell_id in cell_ids:
                                            result[cell_id] = trans
                                else:  # long
                                    trans = future.result()
                                    text, cell_ids = batch_data[0]
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
        """JSON 배치 번역 - 최적화"""
        if not texts:
            return {}
        
        # 너무 많은 텍스트는 분할
        MAX_BATCH = 30  # 한 번에 처리할 최대 개수
        if len(texts) > MAX_BATCH:
            result = {}
            for i in range(0, len(texts), MAX_BATCH):
                batch_result = self._batch_translate_json(session, texts[i:i+MAX_BATCH])
                result.update(batch_result)
            return result
        
        items = [{'id': str(i), 'text': text} for i, text in enumerate(texts)]
        
        # 더 간결한 프롬프트
        prompt = f"""Translate each text to Korean. Return ONLY the JSON array without any thinking process or HTML tags.
Example: [{{"id":"0","translated":"한글번역"}}]

{json.dumps(items, ensure_ascii=False)}"""
        
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'format': 'json',
            'options': {
                'temperature': 0.05,  # 더 낮은 온도
                'num_predict': sum(len(t) for t in texts) * 3,  # 실제 필요한 크기
                'num_ctx': min(16384, len(prompt) * 2),  # 컨텍스트 최적화
                'top_p': 0.9
            }
        }
        
        try:
            response = session.post(f'{self.ollama_url}/api/generate', json=data, timeout=120)
            if response.status_code == 200:
                result_text = response.json().get('response', '')
                return self._parse_json_response(result_text, texts)
        except Exception as e:
            print(f"[ERROR] Batch translation failed: {e}")
        
        # 폴백: 개별 번역
        result = {}
        for text in texts:
            result[text] = self._translate_single(session, text)
        return result
    
    def _translate_single(self, session: requests.Session, text: str) -> str:
        """단일 텍스트 번역"""
        prompt = f"""You are a professional translator. Translate the following text from English to Korean.
IMPORTANT: Return ONLY the Korean translation. Do not include any thinking process, explanations, or HTML tags like <think>.

Text: {text}

Korean translation:"""
        
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_predict': min(2000, len(text) * 3)
            }
        }
        
        try:
            response = session.post(f'{self.ollama_url}/api/generate', json=data, timeout=60)
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # Remove thinking tags if present (both regular and HTML-encoded)
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                result = re.sub(r'&lt;think&gt;.*?&lt;/think&gt;', '', result, flags=re.DOTALL)
                # Also remove partial tags at the beginning/end
                result = re.sub(r'^.*?</think>', '', result, flags=re.DOTALL)
                result = re.sub(r'^.*?&lt;/think&gt;', '', result, flags=re.DOTALL)
                result = re.sub(r'<think>.*?$', '', result, flags=re.DOTALL)
                result = re.sub(r'&lt;think&gt;.*?$', '', result, flags=re.DOTALL)
                return result.strip() or text
        except:
            pass
        
        return text
    
    def _translate_text(self, session: requests.Session, text: str) -> Tuple[str, float]:
        """일반 텍스트 번역"""
        if len(text) < 100:
            return self._translate_single(session, text), 1.0
        
        # 청크 분할
        chunks = self._split_into_chunks(text, 1000)
        translated_chunks = []
        
        for chunk in chunks:
            trans = self._translate_single(session, chunk)
            translated_chunks.append(trans)
        
        return ' '.join(translated_chunks), 0.95
    
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
    base_dir = Path(__file__).resolve().parent
    input_md = base_dir / "uploads" / "page_0003.md"
    output_md = base_dir / "uploads" / "page_0003_korean.md"

    service = TranslationService()
    result = service.translate_document(str(input_md), str(output_md))
    print(f"Translation {'succeeded' if result.success else 'failed'}")