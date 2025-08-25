# services/translation_service.py
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import threading
import requests
import re
import json

try:
    from bs4 import BeautifulSoup  # HTML 표 파싱용
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

# 기존 번역기 임포트
#from quality_contract_translator import QualityContractTranslator

@dataclass
class TranslationResult:
    """번역 결과를 담는 데이터 클래스"""
    success: bool
    input_file: str
    output_file: str
    report: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None

@dataclass
class BatchTranslationResult:
    """배치 번역 결과"""
    success: bool
    results: List[TranslationResult]
    total_files: int
    successful_files: int
    failed_files: int
    average_confidence: float
    total_processing_time: float
    low_quality_files: List[str]

class TranslationService:
    """quality_contract_translator.py의 QualityContractTranslator를 서비스로 래핑"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        

        # 청크 설정 추가 (빠진 부분)
        self.CHUNK_SIZE = 500  # 기본 청크 크기 - 이 줄 추가

        # 번역기 설정
        self.model_name = self.config.get('model_name', 'qwen3:8b')
        self.temperature = self.config.get('temperature', 0.1)
        self.max_retries = self.config.get('max_retries', 3)
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        self.quality_threshold = self.config.get('quality_threshold', 0.6)
        
        # 표 번역 배치 설정 (quality_contract_translator 기준 값 차용)
        self.SHORT_CELL_MAX_CHARS = 50
        self.SHORT_CELL_BATCH_SIZE = 120
        self.MEDIUM_CELL_MAX_CHARS = 200  # 새로 추가
        self.LONG_CELL_MAX_CHARS = 1000  # 새로 추가
        self.MAX_CHARS_PER_BATCH = 2000
        
        # 배치 크기 동적 설정
        self.CELL_BATCH_CONFIGS = {
            'short': {'batch_size': 20, 'num_predict': 500},
            'medium': {'batch_size': 10, 'num_predict': 1000},
            'long': {'batch_size': 3, 'num_predict': 2000},
            'extra_long': {'batch_size': 1, 'num_predict': 4000}
        }


        # 간단한 메모리 캐시 (중복 셀 번역 최소화)
        self._cell_cache: Dict[str, str] = {}
        
        # 스레드별 HTTP 세션만 분리 (번역기는 공유 설정 사용)
        self._local = threading.local()

        #전역 번역기 생성 카운터 (디버깅용?)
        self._session_count = 0
        self._lock = threading.Lock()

        #Ollama 서버 상태 확인
        self._check_ollama_server()

    def _classify_cell(self, text: str) -> str:
        """셀 텍스트 길이에 따른 분류"""
        length = len(text.strip())
        
        if length <= self.SHORT_CELL_MAX_CHARS:
            return 'short'
        elif length <= self.MEDIUM_CELL_MAX_CHARS:
            return 'medium'
        elif length <= self.LONG_CELL_MAX_CHARS:
            return 'long'
        else:
            return 'extra_long'

    def _check_ollama_server(self):
        """ Ollama 서버 상태 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/version",timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama 서버 연결 확인: {self.ollama_url}")
                print(f"📊 공유 모델: {self.model_name}")
            else:
                print(f"⚠️ Ollama 서버 응답 이상: {response.status_code}")
        except Exception as e:
            print(f"❌ Ollama 서버 연결 실패: {e}")
            print(f"   확인사항: ollama serve 실행 여부")
    
    def _get_session(self) -> requests.Session:
        """스레드별 HTTP 세션 획득"""
        
        # 현재 스레드에 세션이 없는 경우에만 생성
        if not hasattr(self._local, 'session') or self._local.session is None:
            
            # 스레드 정보
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
            
            with self._lock:
                self._session_count += 1
                current_count = self._session_count
            
            print(f"\n{'='*50}")
            print(f"🌐 스레드별 HTTP 세션 생성")
            print(f"   스레드 ID: {thread_id}")
            print(f"   스레드 이름: {thread_name}")
            print(f"   세션 순서: {current_count}번째")
            print(f"   Ollama 서버: {self.ollama_url}")
            print(f"   공유 모델: {self.model_name}")
            print(f"{'='*50}")
            
            # 스레드별 독립적인 HTTP 세션 생성
            self._local.session = requests.Session()
            
            # 세션 설정 최적화
            self._local.session.timeout = 300
            self._local.session.headers.update({
                'Content-Type': 'application/json',
                'User-Agent': f'DocumentTranslator-Thread-{thread_id}'
            })
            
            print(f"✅ 스레드 {thread_id}: HTTP 세션 생성 완료!")
            print(f"   세션 ID: {id(self._local.session)}")
            print(f"   타임아웃: 300초")
            print(f"{'='*50}\n")
                
        return self._local.session

    # 5. translate_document 메서드 수정 (로깅 추가)
    def translate_document(self, input_file: str, output_file: str = None) -> TranslationResult:
        """단일 마크다운 문서 번역 (스레드 안전)
            Args:
                input_file: 번역할 마크다운 파일 경로
                output_file: 번역 결과를 저장할 파일 경로 (None이면 자동 생성)
                
            Returns:
                TranslationResult: 번역 결과
        
        """
        thread_id = threading.current_thread().ident
        start_time = time.time()
        print(f"\n📝 스레드 {thread_id}: 번역 시작 - {input_file}")
        
        try:
            # 입력 파일 검증
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")
            
            # 출력 파일 경로 설정
            if output_file is None:
                output_file = str(input_path.parent / f"{input_path.stem}_korean{input_path.suffix}")
            
            self.logger.info(f"스레드 {thread_id}: 문서 번역 시작 - {input_path.name}")

            # 스레드별 독립 HTTP 세션 사용하여 번역 실행
            session = self._get_session()
            report = self._translate_with_session(session,str(input_file), output_file)

            processing_time = time.time() - start_time
            
            # 번역 성공 여부 확인
            success = (
                report.get('average_confidence', 0) >= self.quality_threshold and
                Path(output_file).exists()
            )

            confidence = report.get('average_confidence', 0)
            print(f"✅ 스레드 {thread_id}: 번역 완료!")
            print(f"   처리 시간: {processing_time:.2f}초")
            print(f"   신뢰도: {confidence:.3f}")
            print(f"   성공 여부: {success}")
            
            self.logger.info(f"문서 번역 완료: {processing_time:.2f}초, 신뢰도: {report.get('average_confidence', 0):.2f}")
            
            return TranslationResult(
                success=success,
                input_file=str(input_file),
                output_file=output_file,
                report=report,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ 스레드 {thread_id}: 번역 실패 - {e}")
            self.logger.error(f"스레드 {thread_id}: 문서 번역 실패: {e}")

            return TranslationResult(
                success=False,
                input_file=str(input_file),
                output_file=output_file or "",
                report={},
                processing_time=time.time() - start_time,
                error=str(e)
            )
        
    def _translate_with_session(self, session: requests.Session, input_file: str, output_file: str) -> dict:
        """특정 HTTP 세션을 사용하여 번역 실행"""
        thread_id = threading.current_thread().ident
        
        try:
            # 파일 읽기
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 표 포함 시 전용 파이프라인 사용
            used_table_pipeline = False
            if ('<table' in content.lower()) and BS4_AVAILABLE:
                print(f"🧩 스레드 {thread_id}: HTML 표 감지 → 표 전용 파이프라인 사용")
                final_translation, table_stats = self._translate_markdown_with_tables(session, content)
                sections_count = table_stats.get('sections_count', 0)
                avg_confidence = table_stats.get('avg_confidence', 1.0)
                used_table_pipeline = True
            else:
                # 청크로 분할
                chunks = self._split_into_chunks(content)
                translated_chunks = []
                
                print(f"📄 스레드 {thread_id}: {len(chunks)}개 청크로 분할")
                
                total_confidence = 0.0
                
                for i, chunk in enumerate(chunks, 1):
                    print(f"🔄 스레드 {thread_id}: 청크 {i}/{len(chunks)} 번역 중...")
                    
                    # Ollama API 호출 (독립 세션 사용)
                    translated_text, confidence = self._translate_chunk_ollama(
                        session, chunk, i, len(chunks)
                    )
                    
                    translated_chunks.append(translated_text)
                    total_confidence += confidence
                    
                    # 잠시 대기 (Ollama 서버 부하 분산)
                    time.sleep(0.1)
                
                # 번역 결과 합치기
                final_translation = '\n\n'.join(translated_chunks)
                sections_count = len(chunks)
                avg_confidence = total_confidence / len(chunks) if chunks else 0
            
            # 파일 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_translation)
            
            # 리포트 생성
            return {
                'input_file': input_file,
                'output_file': output_file,
                'sections_count': sections_count,
                'average_confidence': avg_confidence,
                'total_processing_time': time.time(),
                'model_used': self.model_name,
                'thread_id': thread_id,
                'session_id': id(session),
                'used_table_pipeline': used_table_pipeline
            }
            
        except Exception as e:
            print(f"❌ 스레드 {thread_id}: 번역 처리 실패: {e}")
            raise
        
    def _translate_chunk_ollama(self, session: requests.Session, text: str, 
                               idx: int, total: int) -> Tuple[str, float]:
        """Ollama를 사용한 청크 번역 (독립 세션 사용)"""
        
        prompt = f"""당신은 한국어 법무/기술 번역 전문가입니다. 다음 텍스트를 자연스러운 한국어로 번역하세요.

규칙:
- 구조/서식 보존(마크다운/HTML 유지)
- 해설/추론/머리말 금지
- 출력 형식 강제: 최종 번역문만 <md>와 </md> 사이에 넣고, 그 외 텍스트 출력 금지

원문:
{text}

출력:"""
        
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': self.temperature,
                'num_predict': 4000,
                'stop': ['원문:', 'Translation:', '번역:', '\n원문'],
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                # 스레드별 독립 세션 사용
                response = session.post(
                    f'{self.ollama_url}/api/generate',
                    json=data,
                    timeout=300
                )
                response.raise_for_status()
                
                result = response.json()
                translated_text = result.get('response', '').strip()
                
                # 번역 결과 후처리: <md> 블록 추출 및 메타 제거 → 간단 정리
                translated_text = self._strip_meta(translated_text)
                translated_text = self._clean_translation(translated_text)
                
                # 간단한 신뢰도 계산
                confidence = self._calculate_confidence(text, translated_text)
                
                return translated_text, confidence
                
            except Exception as e:
                thread_id = threading.current_thread().ident
                print(f"⚠️ 스레드 {thread_id}: Ollama 요청 실패 (청크 {idx}, 시도 {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    return f"[번역 실패: {str(e)}]", 0.0
                time.sleep(1)  # 재시도 전 대기

    def _clean_translation(self, text: str) -> str:
        """번역 결과 후처리"""
        # 불필요한 패턴 제거
        text = re.sub(r'^번역:\s*', '', text)
        text = re.sub(r'^Translation:\s*', '', text)
        text = re.sub(r'\n번역:\s*', '\n', text)
        
        # 연속된 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    # -------------------- 표/마크다운 전용 처리 --------------------
    def _translate_markdown_with_tables(self, session: requests.Session, content: str) -> Tuple[str, Dict[str, Any]]:
        """마크다운 내 HTML 표 블록을 셀 단위/행 단위 배치 번역 후 재조립.
        표 외 텍스트는 문단 단위로 번역.
        """
        start = time.time()
        parts: List[str] = []
        total_sections = 0
        confidences: List[float] = []

        # 1차: <div> 래핑된 <table> 우선 처리
        pattern = re.compile(r"(<div[\s\S]*?<table[\s\S]*?</table>[\s\S]*?</div>)", re.IGNORECASE)
        cursor = 0
        found_any = False
        for m in pattern.finditer(content):
            found_any = True
            # 표 블록 전의 일반 텍스트 처리
            before = content[cursor:m.start()]
            if before.strip():
                segments = self._split_into_chunks(before, max_chunk_size=1200)
                for seg in segments:
                    res_text, conf = self._translate_chunk_ollama(session, seg, 1, 1)
                    parts.append(res_text)
                    confidences.append(conf)
                    total_sections += 1

            table_block = m.group(1)
            translated_block = self._translate_html_table_block(session, table_block)
            parts.append(translated_block)
            cursor = m.end()

        if not found_any:
            # 2차: 순수 <table> 블록 처리 (wrapper 없음)
            pattern2 = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
            cursor = 0
            for m in pattern2.finditer(content):
                found_any = True
                before = content[cursor:m.start()]
                if before.strip():
                    segments = self._split_into_chunks(before, max_chunk_size=1200)
                    for seg in segments:
                        res_text, conf = self._translate_chunk_ollama(session, seg, 1, 1)
                        parts.append(res_text)
                        confidences.append(conf)
                        total_sections += 1
                table_block = m.group(1)
                translated_block = self._translate_html_table_block(session, table_block)
                parts.append(translated_block)
                cursor = m.end()
            tail = content[cursor:]
            if tail.strip():
                segments = self._split_into_chunks(tail, max_chunk_size=1200)
                for seg in segments:
                    res_text, conf = self._translate_chunk_ollama(session, seg, 1, 1)
                    parts.append(res_text)
                    confidences.append(conf)
                    total_sections += 1
        else:
            # 마지막 꼬리 텍스트 처리
            tail = content[cursor:]
            if tail.strip():
                segments = self._split_into_chunks(tail, max_chunk_size=1200)
                for seg in segments:
                    res_text, conf = self._translate_chunk_ollama(session, seg, 1, 1)
                    parts.append(res_text)
                    confidences.append(conf)
                    total_sections += 1

        elapsed = time.time() - start
        avg_conf = sum(confidences)/len(confidences) if confidences else 1.0
        return ('\n'.join(parts), {"sections_count": total_sections, "avg_confidence": avg_conf, "total_time": elapsed})

    def _translate_html_table_block(self, session: requests.Session, html_block: str) -> str:
        """HTML 표 블록에서 셀 텍스트를 배치 번역하고 치환.
        짧은 셀은 셀 단위, 긴 셀은 JSON 배치 번역.
        """
        if not BS4_AVAILABLE:
            return html_block

        soup = BeautifulSoup(html_block, 'html.parser')
        tables = soup.find_all('table')
        for table in tables:
            try:
                self._normalize_html_table(table)
            except Exception:
                pass
            rows = table.find_all('tr')
            # 헤더 셀 우선 처리
            header_cells = table.find_all(['th'])
            if not header_cells:
                # <th>가 없는 경우 첫 번째 행을 헤더처럼 처리
                first_tr = table.find('tr')
                if first_tr:
                    header_cells = first_tr.find_all('td')
            self._translate_cells_list(session, header_cells, force_cell=True)

            # 본문 셀 처리
            body_rows: List[List] = []
            for tr in rows:
                cells = tr.find_all(['td'])
                if cells:
                    body_rows.append(cells)
            self._translate_table_rows(session, body_rows)

        return str(soup)

    def _normalize_html_table(self, table) -> None:
        """표 구조 정규화: 빈 행 제거, 이어쓰기 병합, 열 개수 패딩."""
        def cells_in_row(tr):
            return tr.find_all(['td', 'th'])

        rows = table.find_all('tr')
        max_cols = 0
        for tr in rows:
            max_cols = max(max_cols, len(cells_in_row(tr)))

        prev_tr = None
        for tr in list(rows):
            cells = cells_in_row(tr)
            texts = [c.get_text(" ", strip=True) for c in cells]
            non_empty_idx = [i for i, t in enumerate(texts) if t]

            # 1) 완전 빈 행 제거
            if not non_empty_idx:
                tr.decompose()
                continue

            # 2) 이어쓰기 병합(휴리스틱)
            if len(non_empty_idx) == 1 and prev_tr is not None:
                only_idx = non_empty_idx[0]
                t = texts[only_idx]
                t_low = t.lower()
                is_continuation = (
                    t.startswith((')', '-', '·', '•')) or
                    t.endswith(')') or
                    any(tok in t_low for tok in ['signer', 'joint', 'continued', 'cont.', 'and'])
                )
                if is_continuation:
                    prev_cells = cells_in_row(prev_tr)
                    prev_texts = [pc.get_text(" ", strip=True) for pc in prev_cells]
                    if any(prev_texts):
                        last_idx = max(i for i, s in enumerate(prev_texts) if s)
                        prev_cells[last_idx].string = (prev_texts[last_idx] + ' ' + t).strip()
                        tr.decompose()
                        continue

            # 3) 열 개수 패딩
            if len(cells) < max_cols:
                for _ in range(max_cols - len(cells)):
                    new_td = table.new_tag('td')
                    tr.append(new_td)

            prev_tr = tr

    def _translate_table_rows(self, session: requests.Session, rows: List[List]) -> None:
        """셀 길이에 따른 적응형 번역"""
    
        # 1. 모든 셀을 길이별로 분류
        cell_buckets = {
            'short': [],
            'medium': [],
            'long': [],
            'extra_long': []
        }

        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                text = (cell.get_text(" ", strip=True) or "").strip()
                if not text:
                    continue
                
                cell_type = self._classify_cell(text)
                cell_buckets[cell_type].append({
                    'cell': cell,
                    'text': text,
                    'row': row_idx,
                    'col': col_idx
                })

        # 2. 각 버킷별로 최적화된 처리
        for cell_type, cells_info in cell_buckets.items():
            if not cells_info:
                continue
            
            config = self.CELL_BATCH_CONFIGS[cell_type]
            
            if cell_type in ['short', 'medium']:
                self._batch_translate_cells(session, cells_info, config)
            elif cell_type == 'long':
                self._translate_long_cells(session, cells_info, config)
            else:  # extra_long
                self._translate_extra_long_cells(session, cells_info, config)

    
    def _translate_cells_list(self, session: requests.Session, cells: List, force_cell: bool) -> None:
        """셀 리스트를 배치 번역(JSON) + 간단 캐시."""
        items: List[Dict[str, str]] = []
        originals: List[str] = []
        for idx, cell in enumerate(cells):
            text = (cell.get_text(" ", strip=True) or "").strip()
            if not text:
                continue
            if not force_cell and len(text) > self.SHORT_CELL_MAX_CHARS:
                continue  # 긴 셀은 행 단위 처리
            key = self._cache_key(text)
            if key in self._cell_cache:
                cell.string = self._cell_cache[key]
            else:
                items.append({"id": f"c{idx}", "text": text})
                originals.append(text)

        # 배치 처리
        pending: List[Dict[str, str]] = []
        acc_len = 0
        def flush_batch(batch: List[Dict[str, str]]):
            if not batch:
                return
            results = self._translate_batch_json(session, batch, fast=True)
            for obj in batch:
                cid = obj["id"]
                original = obj["text"]
                translated = self._strip_meta(results.get(cid) or original) if results else original
                key = self._cache_key(original)
                self._cell_cache[key] = translated

        for obj in items:
            cur_len = len(obj["text"]) + 40
            if (len(pending) >= self.SHORT_CELL_BATCH_SIZE) or (acc_len + cur_len > self.MAX_CHARS_PER_BATCH):
                flush_batch(pending)
                pending = []
                acc_len = 0
            pending.append(obj)
            acc_len += cur_len
        flush_batch(pending)

        # 치환
        for cell in cells:
            text = (cell.get_text(" ", strip=True) or "").strip()
            if not text:
                continue
            if not force_cell and len(text) > self.SHORT_CELL_MAX_CHARS:
                continue
            key = self._cache_key(text)
            if key in self._cell_cache:
                cell.string = self._cell_cache[key]


    def _batch_translate_cells(self, session: requests.Session, cells_info: List[Dict], config: Dict) -> None:
        """짧은/중간 셀 배치 처리"""
        batch_size = config['batch_size']

        for i in range(0, len(cells_info), batch_size):
            batch = cells_info[i:i+batch_size]

            #캐시 확인 및 필터링
            to_translate = []
            for info in batch:
                cache_key = self._cache_key(info['text'])
                if cache_key not in self._cell_cache:
                    to_translate.append(info)
            if not to_translate:
                continue

            # JSON 배치 생성
            items = [{"id":f"c{j}", "text":info['text']} for j, info in enumerate(to_translate)]

            #동적 토큰 계산
            total_chars = sum(len(info['text']) for info in to_translate)
            num_predict = min(total_chars*2, config['num_predict'])

            #번역 실행
            results = self._translate_batch_json_adaptive(session, items, num_predict=num_predict)

            # 결과 적용
            for info, item in zip(to_translate, items):
                translated = results.get(item['id'], info['text'])
                cache_key = self._cache_key(info['text'])
                self._cell_cache[cache_key] = translated
                info['cell'].string = translated

    def _translate_long_cells(self, session: requests.Session, 
                         cells_info: List[Dict], config: Dict) -> None:
        """긴 셀 개별 처리"""
        for info in cells_info:
            text = info['text']
            
            # 리스트 형태 감지
            if '\n' in text and text.count('\n') > 3:
                translated = self._translate_as_list(session, text, config)
            else:
                # 일반 문단으로 처리
                translated, _ = self._translate_chunk_ollama(
                    session, text, 1, 1
                )
            
            info['cell'].string = translated
            time.sleep(0.1)  # 서버 부하 방지
    
    def _translate_extra_long_cells(self, session: requests.Session,
                               cells_info: List[Dict], config: Dict) -> None:
        """매우 긴 셀 청크 처리"""
        for info in cells_info:
            text = info['text']
            chunks = self._smart_chunk(text, chunk_size=self.CHUNK_SIZE)
            
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                # 컨텍스트 유지
                context = ""
                if i > 0 and translated_chunks:
                    # 이전 청크의 마지막 부분을 컨텍스트로
                    last_chunk = translated_chunks[-1]
                    sentences = last_chunk.split('.')
                    if len(sentences) > 1:
                        context = sentences[-2] + '.' if sentences[-2] else ""
                
                prompt = f"""
                {f'이전 문맥: {context}' if context else ''}
                
                다음 내용을 한국어로 번역하세요:
                {chunk}
                
                번역:
                """
                
                translated, _ = self._translate_chunk_ollama(
                    session, prompt, i+1, len(chunks)
                )
                translated_chunks.append(translated)
            
            info['cell'].string = ' '.join(translated_chunks)

    # 유틸리티 메서드 추가
    def _smart_chunk(self, text: str, chunk_size: int = 500) -> List[str]:
        """의미 단위를 보존하는 스마트 청킹"""
        chunks = []
        
        # 줄바꿈이 많으면 줄 단위로 청킹
        if text.count('\n') > 5:
            lines = text.split('\n')
            current_chunk = []
            current_size = 0
            
            for line in lines:
                line_size = len(line)
                if current_size + line_size > chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
        
        else:
            # 문장 단위로 청킹
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sent_size = len(sentence)
                if current_size + sent_size > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sent_size
                else:
                    current_chunk.append(sentence)
                    current_size += sent_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]    

    def _translate_as_list(self, session: requests.Session, 
                       text: str, config: Dict) -> str:
        """리스트 형태 텍스트 최적화 번역"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 5-7개씩 그룹화
        groups = []
        for i in range(0, len(lines), 5):
            groups.append(lines[i:i+5])
        
        translated_groups = []
        for group in groups:
            group_text = '\n'.join(group)
            prompt = f"""
            다음 항목들을 한국어로 번역하세요 (한 줄에 하나씩):
            
            {group_text}
            
            번역:
            """
            
            translated, _ = self._translate_chunk_ollama(
                session, prompt, 1, 1
            )
            translated_groups.append(translated)
        
        return '\n'.join(translated_groups)
    
    def _translate_batch_json_adaptive(self, session: requests.Session, 
                                  items: List[Dict], num_predict: int) -> Dict[str, str]:
        """적응형 JSON 배치 번역"""
        if not items:
            return {}
        
        prompt = f"""
        다음 항목들을 한국어로 번역하세요.
        JSON 배열 형식으로만 응답하세요: [{{"id": "...", "translated": "..."}}]
        
        입력:
        {json.dumps(items, ensure_ascii=False, indent=2)}
        
        출력:
        """
        
        # Ollama 호출 with 동적 설정
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'format': 'json',
            'options': {
                'temperature': 0.1,
                'num_predict': num_predict,
                'num_ctx': 4096  # 컨텍스트 윈도우 확대
            }
        }
        
        try:
            response = session.post(
                f'{self.ollama_url}/api/generate',
                json=data,
                timeout=300
            )
            
            if response.status_code == 200:
                result_text = response.json().get('response', '')
                # JSON 파싱
                result_data = json.loads(result_text)
                return {item['id']: item.get('translated', '') 
                    for item in result_data}
        except Exception as e:
            print(f"배치 번역 실패: {e}")
            
        return {}

    # -------------------- LLM 호출 유틸 --------------------
    def _translate_batch_json(self, session: requests.Session, items: List[Dict[str, str]], fast: bool = True) -> Dict[str, str]:
        if not items:
            return {}
        instruction = (
            "You are a translator. Translate each item's 'text' from English to Korean.\n"
            "Respond ONLY with a compact JSON array: [{\"id\": string, \"translated\": string}, ...].\n"
            "Do not add any explanation."
        )
        examples = json.dumps(items, ensure_ascii=False)
        prompt = f"{instruction}\n\nINPUT:\n{examples}\n\nOUTPUT:"
        resp = self._generate_once_fast(session, prompt) if fast else self._generate_once(session, prompt)
        try:
            m = re.search(r'\[.*\]', resp or '', flags=re.DOTALL)
            payload = m.group(0) if m else (resp or '[]')
            data = json.loads(payload)
            out = {d.get('id'): d.get('translated', '') for d in data if isinstance(d, dict)}
            return out
        except Exception:
            # 폴백: 한 개씩 처리
            out: Dict[str, str] = {}
            for obj in items:
                text = obj["text"]
                single = self._generate_once(
                    session,
                    "Translate to Korean. Return ONLY the translated text, no explanations:\n" +
                    f"{text}\n\nOUTPUT:"
                ) or text
                out[obj["id"]] = self._strip_meta(single)
            return out

    def _generate_once_fast(self, session: requests.Session, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 256
            },
        }
        try:
            response = session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                return raw
        except Exception:
            pass
        return None

    def _generate_once(self, session: requests.Session, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 1024
            },
        }
        try:
            response = session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception:
            pass
        return None

    # -------------------- 메타/캐시 유틸 --------------------
    def _strip_meta(self, text: str) -> str:
        if not text:
            return text
        # think 블록 제거
        t = re.sub(r'(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>', '', text)
        t = re.sub(r'(?is)</\s*think\s*>', '', t)
        # md 블록이 있으면 내부만 사용
        m = re.search(r'(?is)<\s*md\s*>([\s\S]*?)<\s*/\s*md\s*>', t)
        if m:
            t = m.group(1)
        # 앞쪽 영어 해설 제거: 첫 한글/헤더/HTML부터
        try:
            lines = t.splitlines()
            start = 0
            for i, line in enumerate(lines):
                if re.search(r'[가-힣]', line) or line.strip().startswith(('#', '<')):
                    start = i
                    break
            t = "\n".join(lines[start:])
        except Exception:
            pass
        return t.strip()

    def _cache_key(self, text: str) -> str:
        norm = re.sub(r"\s+", " ", text.strip())
        try:
            import hashlib
            return hashlib.sha1(norm.encode("utf-8")).hexdigest()
        except Exception:
            return norm
    
    def _calculate_confidence(self, original: str, translated: str) -> float:
        """번역 신뢰도 계산"""
        if not translated or translated.startswith('[번역 실패'):
            return 0.0
        
        # 길이 기반 신뢰도
        length_ratio = len(translated) / max(len(original), 1)
        if length_ratio < 0.3 or length_ratio > 3.0:
            return 0.3
        
        # 한국어 문자 비율
        korean_chars = sum(1 for char in translated if '가' <= char <= '힣')
        korean_ratio = korean_chars / max(len(translated), 1)
        
        # 전체 신뢰도 계산
        confidence = min(0.9, 0.7 + korean_ratio * 0.2)
        return confidence
    
    def _split_into_chunks(self, content: str, max_chunk_size: int = 2000) -> list[str]:
        """텍스트를 청크로 분할"""
        # 문단으로 분할
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]

    # 6. 새로 추가할 디버깅 메서드
    def get_thread_info(self) -> Dict[str, Any]:
        """ 현재 스레드의 번역기 정보 반환(디버깅용)"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        has_session = hasattr(self._local, 'session') and self._local.session is not None

        return {
            "thread_id": thread_id,
            "thread_name": thread_name,
            "has_session": has_session,
            "total_sessions_created":self._session_count,
            "session_id":id(self._local.session) if has_session else None,
            "ollama_model": self.model_name,
            "ollama_url": self.ollama_url
        } 
    def translate_documents_batch(self, markdown_files: List[str], output_dir: str = None) -> BatchTranslationResult:
        """
        여러 마크다운 파일 배치 번역
        
        Args:
            markdown_files: 번역할 마크다운 파일 경로 리스트
            output_dir: 번역 결과를 저장할 디렉토리 (None이면 각 파일과 같은 디렉토리)
            
        Returns:
            BatchTranslationResult: 배치 번역 결과
        """
        start_time = time.time()
        results = []
        successful_files = 0
        failed_files = 0
        total_confidence = 0.0
        confidence_count = 0
        low_quality_files = []
        
        self.logger.info(f"배치 번역 시작: {len(markdown_files)}개 파일")
        
        for i, md_file in enumerate(markdown_files, 1):
            self.logger.info(f"번역 진행 중 ({i}/{len(markdown_files)}): {Path(md_file).name}")
            
            try:
                # 출력 파일 경로 결정
                input_path = Path(md_file)
                if output_dir:
                    output_path = Path(output_dir) / f"{input_path.stem}_korean{input_path.suffix}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_file = str(output_path)
                else:
                    output_file = None
                
                # 개별 파일 번역
                result = self.translate_document(md_file, output_file)
                results.append(result)
                
                if result.success:
                    successful_files += 1
                    confidence = result.report.get('average_confidence', 0)
                    total_confidence += confidence
                    confidence_count += 1
                    
                    # 품질이 낮은 파일 기록
                    if confidence < self.quality_threshold:
                        low_quality_files.append(md_file)
                else:
                    failed_files += 1
                    
            except Exception as e:
                self.logger.error(f"파일 번역 중 예외 발생 ({md_file}): {e}")
                failed_files += 1
                results.append(TranslationResult(
                    success=False,
                    input_file=md_file,
                    output_file="",
                    report={},
                    processing_time=0,
                    error=str(e)
                ))
        
        # 전체 결과 계산
        total_processing_time = time.time() - start_time
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        batch_success = (failed_files == 0) and (len(low_quality_files) == 0)
        
        self.logger.info(f"배치 번역 완료: {successful_files}/{len(markdown_files)} 성공, "
                        f"평균 신뢰도: {average_confidence:.2f}, "
                        f"총 시간: {total_processing_time:.2f}초")
        
        return BatchTranslationResult(
            success=batch_success,
            results=results,
            total_files=len(markdown_files),
            successful_files=successful_files,
            failed_files=failed_files,
            average_confidence=average_confidence,
            total_processing_time=total_processing_time,
            low_quality_files=low_quality_files
        )
    
    def retry_failed_translations(self, batch_result: BatchTranslationResult) -> BatchTranslationResult:
        """
        실패한 번역 재시도
        
        Args:
            batch_result: 이전 배치 번역 결과
            
        Returns:
            BatchTranslationResult: 재시도 결과
        """
        failed_files = [r.input_file for r in batch_result.results if not r.success]
        
        if not failed_files:
            self.logger.info("재시도할 실패 파일이 없습니다.")
            return batch_result
        
        self.logger.info(f"실패한 번역 재시도: {len(failed_files)}개 파일")
        
        # 재시도 시 설정 조정 (온도 낮추기, 재시도 횟수 증가)
        original_config = self.config.copy()
        self.config['temperature'] = max(0.05, self.config.get('temperature', 0.1) - 0.05)
        self.config['max_retries'] = self.config.get('max_retries', 3) + 2
        
        # 번역기 재초기화
        self._translator = None
        
        try:
            retry_result = self.translate_documents_batch(failed_files)
            
            # 성공한 재시도 결과로 원본 결과 업데이트
            updated_results = []
            retry_dict = {r.input_file: r for r in retry_result.results}
            
            for original_result in batch_result.results:
                if original_result.input_file in retry_dict:
                    retry_res = retry_dict[original_result.input_file]
                    if retry_res.success:
                        updated_results.append(retry_res)
                    else:
                        updated_results.append(original_result)  # 여전히 실패
                else:
                    updated_results.append(original_result)  # 원래 성공했던 것
            
            # 통계 재계산
            successful = sum(1 for r in updated_results if r.success)
            failed = len(updated_results) - successful
            
            avg_conf = sum(r.report.get('average_confidence', 0) 
                          for r in updated_results if r.success) / max(successful, 1)
            
            low_quality = [r.input_file for r in updated_results 
                          if r.success and r.report.get('average_confidence', 0) < self.quality_threshold]
            
            return BatchTranslationResult(
                success=(failed == 0 and len(low_quality) == 0),
                results=updated_results,
                total_files=len(updated_results),
                successful_files=successful,
                failed_files=failed,
                average_confidence=avg_conf,
                total_processing_time=batch_result.total_processing_time + retry_result.total_processing_time,
                low_quality_files=low_quality
            )
            
        finally:
            # 원래 설정 복원
            self.config = original_config
            
    
    def get_translation_status(self, input_files: List[str], output_dir: str = None) -> Dict[str, Dict[str, Any]]:
        """
        번역 상태 확인
        
        Args:
            input_files: 확인할 입력 파일 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            Dict: 파일별 번역 상태
        """
        status = {}
        
        for input_file in input_files:
            input_path = Path(input_file)
            
            # 출력 파일 경로 추정
            if output_dir:
                output_path = Path(output_dir) / f"{input_path.stem}_korean{input_path.suffix}"
            else:
                output_path = input_path.parent / f"{input_path.stem}_korean{input_path.suffix}"
            
            file_status = {
                'input_exists': input_path.exists(),
                'output_exists': output_path.exists(),
                'input_size': input_path.stat().st_size if input_path.exists() else 0,
                'output_size': output_path.stat().st_size if output_path.exists() else 0,
                'needs_translation': not output_path.exists() or output_path.stat().st_mtime < input_path.stat().st_mtime
            }
            
            status[str(input_file)] = file_status
        
        return status
    
    def cleanup_temp_files(self, output_dir: str):
        """임시 파일 정리 - 디스크의 불필요한 파일 삭제"""
        try:
            output_path = Path(output_dir)
            if output_path.exists():
                # raw 파일들 정리
                for raw_file in output_path.glob("*_raw.md"):
                    raw_file.unlink()
                    self.logger.debug(f"임시 파일 삭제: {raw_file}")
                
                # 캐시 파일 정리
                cache_file = output_path / "translation_cache.json"
                if cache_file.exists():
                    cache_file.unlink()
                    self.logger.debug(f"캐시 파일 삭제: {cache_file}")
                
                self.logger.info(f"임시 파일 정리 완료: {output_dir}")
        except Exception as e:
            self.logger.warning(f"임시 파일 정리 실패: {e}")
    
    def cleanup(self):
        """리소스 정리 - 메모리/네트워크 리소스 해제"""
        if hasattr(self._local, 'session') and self._local.session:
            self._local.session.close()
            self._local.session = None
            self.logger.info("HTTP 세션 정리 완료")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            self.cleanup()
        except:
            pass