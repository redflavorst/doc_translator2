import requests
import json
import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import html as html_mod

try:
    from bs4 import BeautifulSoup  # HTML 표 파싱용
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """번역 결과를 저장하는 데이터 클래스"""
    original_text: str
    translated_text: str
    confidence_score: float
    content_type: str
    processing_time: float

class LegalTerminologyDatabase:
    """법률 용어 데이터베이스"""
    
    def __init__(self):
        self.legal_terms = {
            # 회사명 (영문 유지)
            "Kimberly-Clark": "Kimberly-Clark",
            "K-C": "K-C",
            "YUHAN-KIMBERLY": "YUHAN-KIMBERLY",
            "Global Systems": "Global Systems",
            
            # 제품 관련 용어
            "Licensed Products": "라이선스 제품",
            "Consumer Products": "소비자 제품",
            "Professional Market": "전문시장",
            "Business-to-Business": "기업간 거래",
            "Disposable": "일회용",
            "Nonwovens": "부직포",
            "Industrial": "산업용",
        }
        
        self.legal_phrases = {
            "in lieu of": "~을 대신하여",
            "without limitation": "제한 없이",
            "subject to": "~에 따라",
            "provided that": "단, ~인 경우",
            "notwithstanding": "~에도 불구하고",
            "herein": "본 계약서에서",
            "hereof": "본 계약서의",
            "hereby": "이로써",
            "heretofore": "지금까지",
            "hereafter": "이후",
            "whereas": "반면에",
            "wherefore": "따라서",
            "to the extent": "~하는 범위에서",
            "from time to time": "수시로",
            "best efforts": "최선의 노력",
            "commercially reasonable": "상업적으로 합리적인",
            "good faith": "선의로",
            "arm's length": "독립당사자간",
        }
    
    def add_legal_terms(self, terms_dict: Dict[str, str]):
        """법률 용어 추가"""
        self.legal_terms.update(terms_dict)
        
    def add_legal_phrases(self, phrases_dict: Dict[str, str]):
        """법률 구문 추가"""
        self.legal_phrases.update(phrases_dict)
        
    def get_all_terms(self):
        """모든 용어 반환"""
        return {**self.legal_terms, **self.legal_phrases}

class QualityContractTranslator:
    """고품질 계약서 번역기"""
    
    def __init__(self, 
                 model_name: str = "qwen2.5:14b",
                 ollama_url: str = "http://localhost:11434",
                 max_retries: int = 3,
                 temperature: float = 0.2,
                 debug_raw_output: bool = True):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.debug_raw_output = debug_raw_output  # 원본 응답을 그대로 파일에 남기는 디버그 모드
        self._raw_outputs = []  # 디버그용 응답 수집 버퍼
        self.terminology_db = LegalTerminologyDatabase()
        self.session = requests.Session()  # 연결 재사용
        self.cache_path = Path("translation_cache.json")
        self.cache: Dict[str, str] = self._load_cache()
        
        # 품질 확보를 위한 설정
        self.quality_settings = {
            "temperature": temperature,      # 일관성 확보
            "top_p": 0.85,                  # 적당한 다양성
            "repeat_penalty": 1.1,          # 반복 방지
            "num_predict": 3000,            # 충분한 토큰
            "stop": ["</translation>", "번역 완료"]  # 불필요한 출력 방지
        }

        # 표 번역 설정
        # 배치/길이 기준 (성능 최적화)
        self.SHORT_CELL_MAX_CHARS = 90          # 짧은 셀 임계값(문자)
        self.SHORT_CELL_BATCH_SIZE = 120        # 배치 최대 아이템 수(짧은 셀)
        self.MAX_CHARS_PER_BATCH = 3500         # 배치 총 길이 상한(문자)
        self.LONG_CELL_ROW_BATCH = 2            # (이전 로직 잔재) 사용 안함; 긴 셀도 JSON 배치로 처리
    
    def create_premium_prompt(self, text: str, content_type: str) -> str:
        """최고 품질의 번역 프롬프트 생성"""
        
        # 용어집 문맥 제공
        relevant_terms = self._extract_relevant_terms(text)
        term_context = ""
        if relevant_terms:
            term_context = "\n핵심 용어 번역 가이드:\n"
            for eng, kor in relevant_terms.items():
                term_context += f"- {eng} → {kor}\n"
        
        base_prompt = f"""당신은 대한민국 최고 수준의 법무 번역 전문가입니다. 다음 영어 법률 문서를 한국어로 번역해주세요.

번역 원칙:
1. 법률적 정확성: 한국 법률 용어와 관례에 맞게 번역
2. 일관성: 동일한 용어는 문서 전체에서 동일하게 번역
3. 명확성: 의미가 명확하고 자연스러운 한국어 표현 사용
4. 격식: 공식적인 계약서 문체 유지
5. 구조 보존: 마크다운 형식과 조항 번호 유지
6. 내부 추론/설명 금지: 해설, reasoning, think/analysis 섹션을 출력하지 말 것
7. 출력 형식 강제: 최종 번역문만 <md>와 </md> 사이에 넣어 출력하고, 그 외 어떤 텍스트도 출력하지 말 것

번역 지침:
- 회사명과 고유명사는 영문 그대로 유지
- 법률 용어는 표준 한국어 법률 용어 사용
- 문맥상 자연스럽게 의역하되 의미 변경 금지
- 숫자, 날짜, 참조 조항은 정확히 유지
- HTML 태그가 있다면 구조 그대로 유지
{term_context}
번역 대상 텍스트:
{text}

한국어 번역 (마크다운 형식 유지). 아래 형식으로만 출력:

<md>
...여기에 번역 마크다운만...
</md>
"""

        return base_prompt
    
    def _extract_relevant_terms(self, text: str) -> Dict[str, str]:
        """텍스트에서 관련 용어 추출"""
        relevant_terms = {}
        text_lower = text.lower()
        
        # 정확한 매칭을 위해 단어 경계 확인
        for eng_term, kor_term in self.terminology_db.legal_terms.items():
            if re.search(r'\b' + re.escape(eng_term.lower()) + r'\b', text_lower):
                relevant_terms[eng_term] = kor_term
        
        return relevant_terms
    
    def translate_with_quality_check(self, text: str) -> TranslationResult:
        """품질 확인이 포함된 번역"""
        start_time = time.time()
        content_type = self._detect_content_type(text)
        
        best_translation = None
        highest_confidence = 0.0
        
        for attempt in range(self.max_retries):
            logger.info(f"번역 시도 {attempt + 1}/{self.max_retries}")
            
            # 각 시도마다 약간씩 다른 설정 사용 (다양성 확보)
            temp_adjustment = attempt * 0.05
            current_temp = min(self.temperature + temp_adjustment, 0.4)
            
            translation = self._single_translation_attempt(text, content_type, current_temp)
            
            if translation:
                confidence = self._calculate_confidence_score(text, translation, content_type)
                logger.info(f"시도 {attempt + 1} 신뢰도: {confidence:.2f}")
                
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_translation = translation
                
                # 조기 종료 기준: 어느 시도에서든 신뢰도가 0.60 이상이면 종료
                if confidence >= 0.60:
                    logger.info("높은 품질 번역 달성, 조기 종료")
                    break
            
            # 재시도 간 간격
            if attempt < self.max_retries - 1:
                time.sleep(1)
        
        processing_time = time.time() - start_time
        
        if not best_translation:
            logger.error("모든 번역 시도 실패")
            best_translation = f"[번역 실패] {text}"
            highest_confidence = 0.0
        
        return TranslationResult(
            original_text=text,
            translated_text=best_translation,
            confidence_score=highest_confidence,
            content_type=content_type,
            processing_time=processing_time
        )
    
    def _single_translation_attempt(self, text: str, content_type: str, temperature: float) -> Optional[str]:
        """단일 번역 시도"""
        prompt = self.create_premium_prompt(text, content_type)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                **self.quality_settings,
                "temperature": temperature
            }
        }
        
        try:
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=180  # 품질을 위해 넉넉한 시간 할당
            )
            
            if response.status_code == 200:
                result = response.json()
                translation = result.get('response', '').strip()
                if self.debug_raw_output:
                    # 필터/정리 없이 원본 그대로 반환
                    return translation
                return self._clean_and_validate_translation(translation, text)
            else:
                logger.error(f"API 에러: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"번역 에러: {e}")
            return None
    
    def _clean_and_validate_translation(self, translation: str, original: str) -> str:
        """번역 결과 정리 및 검증"""
        # HTML 엔티티(예: &lt;think&gt;) 복원 후 정리
        try:
            translation = html_mod.unescape(translation)
        except Exception:
            pass
        # 프롬프트/사전 설명 제거: 모델에게 <md>...</md>로만 출력하도록 강제했으므로, 
        # 가드로 <md> 블록만 추출. 없을 경우 기존 규칙 적용.
        md_match = re.search(r'(?is)<\s*md\s*>([\s\S]*?)<\s*/\s*md\s*>', translation)
        if md_match:
            translation = md_match.group(1)
        else:
            # 백업 정리 루틴: think/머리말 제거
            translation = re.sub(r'(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>', '', translation)
            try:
                m = re.search(r'(?is)</\s*think\s*>', translation)
                if m:
                    prefix = translation[:m.end()]
                    if not re.search(r'[가-힣]', prefix) and not re.search(r'^[#<]', prefix, re.MULTILINE):
                        translation = translation[m.end():]
                    else:
                        translation = re.sub(r'(?is)</\s*think\s*>', '', translation)
            except Exception:
                pass
        # 2) "번역:"/"Translation:" 류의 머리말 제거
        translation = re.sub(r'^.*?한국어 번역.*?:', '', translation, flags=re.DOTALL)
        translation = re.sub(r'^.*?번역.*?:', '', translation, flags=re.DOTALL)
        translation = re.sub(r'^.*?Translation.*?:', '', translation, flags=re.DOTALL | re.IGNORECASE)
        
        # 불필요한 설명 제거
        translation = re.sub(r'\[.*?번역.*?\]', '', translation)
        translation = re.sub(r'\(.*?번역.*?\)', '', translation)
        
        # 4) 선행 영어 해설/메타 제거: 첫 유효 본문(헤더/한글/HTML 태그)부터 유지
        try:
            lines = translation.splitlines()
            start_idx = 0
            for i, line in enumerate(lines):
                if re.search(r'[가-힣]', line) or line.strip().startswith(('#', '<')):
                    start_idx = i
                    break
            translation = "\n".join(lines[start_idx:])
        except Exception:
            pass

        # 5) 잔여 토큰 제거
        translation = re.sub(r'(?is)</\s*think\s*>', '', translation)

        # 6) 과도한 줄바꿈 정리
        translation = re.sub(r'\n\s*\n\s*\n+', '\n\n', translation)
        
        # HTML 구조 검증
        if '<table' in original and '<table' not in translation:
            logger.warning("⚠️ 테이블 구조가 손실될 수 있습니다")
        
        # 마크다운 헤더 검증
        original_headers = re.findall(r'^#+\s+.*$', original, re.MULTILINE)
        translated_headers = re.findall(r'^#+\s+.*$', translation, re.MULTILINE)
        
        if len(original_headers) != len(translated_headers):
            logger.warning("⚠️ 마크다운 헤더 구조가 변경되었을 수 있습니다")
        
        # 용어 일관성 적용
        translation = self._apply_terminology_consistency(translation)
        
        return translation.strip()
    
    def _apply_terminology_consistency(self, text: str) -> str:
        """용어 일관성 적용"""
        # 회사명 보정
        company_names = ["Kimberly-Clark", "K-C", "YUHAN-KIMBERLY", "Global Systems"]
        for name in company_names:
            # 한국어 조사와 함께 사용된 경우 원문 유지
            text = re.sub(f'{name}\\s*[를을는이가와과의에게서로]', 
                         lambda m: name + m.group(0)[len(name):], text)
        
        # 표준 법률 용어 적용
        term_corrections = {
            "지적 재산권": "지적재산권",
            "지적 재산": "지적재산",
            "라이센스": "라이선스",
            "라이센싱": "라이선싱",
            "계약 당사자": "계약당사자",
            "제 3자": "제3자",
            "제삼자": "제3자",
        }
        
        for wrong, correct in term_corrections.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _detect_content_type(self, text: str) -> str:
        """컨텐츠 유형 감지"""
        text_lower = text.lower()
        
        if '<table' in text_lower or '<div' in text_lower:
            return 'table_content'
        elif any(word in text_lower for word in ['article', 'section', 'agreement', 'warranty', 'license']):
            return 'legal_article'
        elif any(word in text_lower for word in ['appendix', 'exhibit', 'schedule']):
            return 'appendix'
        elif any(word in text_lower for word in ['product', 'material', 'disposable', 'industrial']):
            return 'product_specification'
        else:
            return 'general_legal'
    
    def _calculate_confidence_score(self, original: str, translation: str, content_type: str) -> float:
        """번역 품질 신뢰도 계산"""
        score = 0.0
        
        # 기본 점수 (완전한 번역인 경우)
        if translation and not translation.startswith('[번역 실패]'):
            score += 0.3
        
        # 길이 적절성 (원문의 0.5~2배 범위)
        if original and translation:
            length_ratio = len(translation) / len(original)
            if 0.5 <= length_ratio <= 2.0:
                score += 0.2
            elif 0.3 <= length_ratio <= 3.0:
                score += 0.1
        
        # 구조 보존 점수
        original_structure_score = self._check_structure_preservation(original, translation)
        score += original_structure_score * 0.2
        
        # 용어 일관성 점수
        terminology_score = self._check_terminology_consistency(original, translation)
        score += terminology_score * 0.2
        
        # 한국어 자연스러움 점수
        fluency_score = self._check_korean_fluency(translation)
        score += fluency_score * 0.1
        
        return min(score, 1.0)
    
    def _check_structure_preservation(self, original: str, translation: str) -> float:
        """구조 보존 확인"""
        score = 0.0
        
        # 마크다운 헤더 확인
        orig_headers = len(re.findall(r'^#+', original, re.MULTILINE))
        trans_headers = len(re.findall(r'^#+', translation, re.MULTILINE))
        if orig_headers > 0 and orig_headers == trans_headers:
            score += 0.3
        
        # HTML 태그 확인
        orig_html = len(re.findall(r'<[^>]+>', original))
        trans_html = len(re.findall(r'<[^>]+>', translation))
        if orig_html > 0 and orig_html == trans_html:
            score += 0.3
        
        # 번호 매기기 확인
        orig_numbers = len(re.findall(r'^\d+\.\d+', original, re.MULTILINE))
        trans_numbers = len(re.findall(r'^\d+\.\d+', translation, re.MULTILINE))
        if orig_numbers > 0 and orig_numbers == trans_numbers:
            score += 0.4
        
        return min(score, 1.0)
    
    def _check_terminology_consistency(self, original: str, translation: str) -> float:
        """용어 일관성 확인"""
        score = 0.0
        consistent_terms = 0
        total_terms = 0
        
        for eng_term, kor_term in self.terminology_db.legal_terms.items():
            if eng_term.lower() in original.lower():
                total_terms += 1
                if kor_term in translation:
                    consistent_terms += 1
        
        if total_terms > 0:
            score = consistent_terms / total_terms
        else:
            score = 1.0  # 특별한 용어가 없으면 만점
        
        return score
    
    def _check_korean_fluency(self, translation: str) -> float:
        """한국어 자연스러움 확인"""
        score = 1.0
        
        # 비자연스러운 패턴 감지
        unnatural_patterns = [
            r'[a-zA-Z]+\s+[을를이가는]',  # 영어 단어 + 조사 (부자연스러운 조합)
            r'것입니다\s+것입니다',       # 중복 표현
            r'에\s+에\s+',                # 조사 중복
            r'을\s+을\s+',                # 조사 중복
        ]
        
        for pattern in unnatural_patterns:
            if re.search(pattern, translation):
                score -= 0.2
        
        return max(score, 0.0)
    
    def translate_document_premium(self, input_file: str, output_file: str = None, postclean_think: bool = True) -> Dict:
        """프리미엄 품질로 전체 문서 번역"""
        logger.info(f"프리미엄 번역 시작: {input_file}")
        
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # HTML 표가 포함된 페이지는 표/텍스트를 분리 처리
        if '<table' in content.lower() and BS4_AVAILABLE:
            logger.info("HTML 표 감지 → 셀 배치 번역 파이프라인 사용")
            translated, table_stats = self._translate_markdown_with_tables(content)
            final_translation = self._final_post_processing(translated)
            sections_count = table_stats.get('sections_count', 0)
            avg_confidence = table_stats.get('avg_confidence', 1.0)
            total_time = table_stats.get('total_time', 0.0)
        else:
            # 지능적 섹션 분할 (문단/헤더 기준)
            sections = self._intelligent_text_splitting(content)
            logger.info(f"총 {len(sections)}개 섹션으로 분할")
            
            translation_results = []
            total_confidence = 0.0
            
            for i, section in enumerate(sections, 1):
                logger.info(f"섹션 {i}/{len(sections)} 번역 중...")
                result = self.translate_with_quality_check(section)
                translation_results.append(result)
                total_confidence += result.confidence_score
                logger.info(f"섹션 {i} 완료 - 신뢰도: {result.confidence_score:.2f}, "
                           f"처리시간: {result.processing_time:.1f}초")
                if result.confidence_score < 0.6:
                    logger.warning(f"⚠️ 섹션 {i}의 번역 품질이 낮습니다 (신뢰도: {result.confidence_score:.2f})")
            
            # 결과 합치기
            final_translation = '\n\n'.join([r.translated_text for r in translation_results])
            final_translation = self._final_post_processing(final_translation)
            sections_count = len(sections)
            avg_confidence = total_confidence / len(translation_results) if translation_results else 0
            total_time = sum(r.processing_time for r in translation_results)
        
        # 출력 파일명 설정
        if not output_file:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_premium_korean{input_path.suffix}"
        
        # think 블록 후처리 (옵션) + <md> 태그 제거
        to_write = self._remove_think_blocks(final_translation) if postclean_think else final_translation
        to_write = self._strip_md_tags(to_write)

        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(to_write)
        # 디버그: 원본(필터 전) 응답을 별도 마크다운으로 저장
        if self.debug_raw_output:
            raw_path = Path(str(output_file).replace(str(Path(output_file).suffix), ''))
            raw_path = raw_path.with_name(raw_path.name + "_raw").with_suffix(Path(output_file).suffix)
            try:
                with open(raw_path, 'w', encoding='utf-8') as rf:
                    rf.write(final_translation)
            except Exception:
                pass
        
        report = {
            "input_file": input_file,
            "output_file": str(output_file),
            "sections_count": sections_count,
            "average_confidence": avg_confidence,
            "total_processing_time": total_time,
            "model_used": self.model_name,
            "quality_settings": self.quality_settings,
            "low_quality_sections": []
        }
        
        logger.info(f"번역 완료!")
        logger.info(f"출력 파일: {output_file}")
        logger.info(f"평균 신뢰도: {avg_confidence:.2f}")
        logger.info(f"총 처리시간: {total_time:.1f}초")
        
        return report

    # -------------------- Think 블록 제거 도구 --------------------
    def _remove_think_blocks(self, text: str) -> str:
        """<think>...</think> 및 &lt;think&gt; 형태까지 제거. 고아 토큰도 제거."""
        try:
            text = html_mod.unescape(text)
        except Exception:
            pass
        text = re.sub(r'(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>', '', text)
        text = re.sub(r'(?is)<\s*think\s*>', '', text)
        text = re.sub(r'(?is)</\s*think\s*>', '', text)
        return text

    def _strip_md_tags(self, text: str) -> str:
        """<md>...</md> 래퍼 태그 제거 (&lt;md&gt; 지원)."""
        try:
            text = html_mod.unescape(text)
        except Exception:
            pass
        # 태그만 제거하고 내부 본문은 유지
        text = re.sub(r'(?is)</?\s*md\s*>', '', text)
        return text

    # -------------------- 표/마크다운 전용 처리 --------------------
    def _translate_markdown_with_tables(self, content: str) -> Tuple[str, Dict]:
        """마크다운 내 HTML 표 블록을 셀 단위 배치 번역 후 재조립.
        표 외 텍스트는 문단 단위로 번역.
        """
        start = time.time()
        parts: List[str] = []
        total_sections = 0
        confidences: List[float] = []

        # 표 블록 기준으로 split (div 내부 table 우선 처리)
        pattern = re.compile(r"(<div[\s\S]*?<table[\s\S]*?</table>[\s\S]*?</div>)", re.IGNORECASE)
        cursor = 0
        for m in pattern.finditer(content):
            # 표 블록 전의 일반 텍스트 처리
            before = content[cursor:m.start()]
            if before.strip():
                segments = self._split_long_section(before)
                for seg in segments:
                    res = self.translate_with_quality_check(seg)
                    parts.append(res.translated_text)
                    confidences.append(res.confidence_score)
                    total_sections += 1

            table_block = m.group(1)
            translated_block = self._translate_html_table_block(table_block)
            parts.append(translated_block)
            cursor = m.end()

        # 마지막 꼬리 텍스트 처리
        tail = content[cursor:]
        if tail.strip():
            segments = self._split_long_section(tail)
            for seg in segments:
                res = self.translate_with_quality_check(seg)
                parts.append(res.translated_text)
                confidences.append(res.confidence_score)
                total_sections += 1

        elapsed = time.time() - start
        avg_conf = sum(confidences)/len(confidences) if confidences else 1.0
        return ('\n'.join(parts), {"sections_count": total_sections, "avg_confidence": avg_conf, "total_time": elapsed})

    def _translate_html_table_block(self, html_block: str) -> str:
        """HTML 표 블록에서 셀 텍스트를 배치 번역하고 치환.
        짧은 셀은 셀 단위, 긴 셀은 행 단위로 번역.
        """
        if not BS4_AVAILABLE:
            return html_block

        soup = BeautifulSoup(html_block, 'html.parser')
        tables = soup.find_all('table')
        for table in tables:
            # 번역 전에 구조 정규화(빈 행 제거, 이어쓰기 행 병합 등)
            try:
                self._normalize_html_table(table)
            except Exception:
                pass
            rows = table.find_all('tr')
            # 헤더 셀 우선 처리
            header_cells = table.find_all(['th'])
            self._translate_cells_list(header_cells, force_cell=True)

            # 본문 셀 처리
            body_rows: List[List] = []
            for tr in rows:
                cells = tr.find_all(['td'])
                if cells:
                    body_rows.append(cells)
            self._translate_table_rows(body_rows)

        # 포맷 보존을 위해 원래 블록의 외곽 div/html/body 태그 유지
        return str(soup)

    def _normalize_html_table(self, table) -> None:
        """표 구조 정규화.
        - 완전 빈 행 제거
        - '이어쓰기'로 보이는 단일 셀 행을 직전 행의 마지막 비어있지 않은 셀에 합침
        - 각 행을 동일한 열 개수로 패딩
        """
        # 열 개수 추정 (헤더 제외하지 않고 최대값)
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

            # 2) 이어쓰기 행 병합(휴리스틱): 한 셀만 채워져 있고, 그 셀 텍스트가 괄호/접속사/잇는 표현이면 이전 행에 붙임
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
                    # 이전 행의 마지막 비어있지 않은 셀을 찾아 이어붙임
                    prev_cells = cells_in_row(prev_tr)
                    prev_texts = [pc.get_text(" ", strip=True) for pc in prev_cells]
                    if any(prev_texts):
                        last_idx = max(i for i, s in enumerate(prev_texts) if s)
                        prev_cells[last_idx].string = (prev_texts[last_idx] + ' ' + t).strip()
                        tr.decompose()
                        continue

            # 3) 열 개수 패딩(렌더 안정용)
            if len(cells) < max_cols:
                for _ in range(max_cols - len(cells)):
                    new_td = table.new_tag('td')
                    tr.append(new_td)

            prev_tr = tr

    def _translate_table_rows(self, rows: List[List]) -> None:
        """셀 길이에 따라 하이브리드 번역 적용."""
        # 1) 짧은 셀 모아서 배치 번역(아이템 수 + 총 길이 동시 제약)
        short_cells = []
        for r in rows:
            for cell in r:
                text = (cell.get_text(" ", strip=True) or "").strip()
                if not text:
                    continue
                if len(text) <= self.SHORT_CELL_MAX_CHARS:
                    short_cells.append((cell, text))

        self._translate_cells_list([c for c, _ in short_cells], force_cell=False)

        # 2) 긴 셀은 JSON 배치 번역(추론/설명 금지, 포맷 강제)
        long_cells: List[Tuple] = []
        for r in rows:
            for cell in r:
                text = (cell.get_text(" ", strip=True) or "").strip()
                if text and len(text) > self.SHORT_CELL_MAX_CHARS:
                    long_cells.append((cell, text))

        if long_cells:
            items = [{"id": f"L{i}", "text": t} for i, (_, t) in enumerate(long_cells)]
            # 길이/개수 제약에 맞춰 묶음 생성
            pending = []
            acc_len = 0
            def flush_batch(batch):
                if not batch:
                    return
                res_map = self._translate_batch_json(batch, fast=True)
                for obj in batch:
                    cid = obj["id"]
                    translated = self._strip_meta(res_map.get(cid) or obj["text"])
                    idx = int(cid[1:])  # L{index}
                    cell = long_cells[idx][0]
                    cell.string = translated

            for obj in items:
                cur_len = len(obj["text"]) + 40  # JSON 오버헤드 여유치
                if (len(pending) >= self.SHORT_CELL_BATCH_SIZE) or (acc_len + cur_len > self.MAX_CHARS_PER_BATCH):
                    flush_batch(pending)
                    pending = []
                    acc_len = 0
                pending.append(obj)
                acc_len += cur_len
            flush_batch(pending)

    def _translate_cells_list(self, cells: List, force_cell: bool) -> None:
        """셀 리스트를 배치 번역(JSON) + 캐시.
        force_cell=True 이면 모두 개별 처리(헤더 등).
        """
        items = []
        for idx, cell in enumerate(cells):
            text = (cell.get_text(" ", strip=True) or "").strip()
            if not text:
                continue
            if not force_cell and len(text) > self.SHORT_CELL_MAX_CHARS:
                continue  # 긴 셀은 행 단위 처리
            key = self._cache_key(text)
            if key in self.cache:
                cell.string = self.cache[key]
            else:
                items.append({"id": f"c{idx}", "text": text})

        # 배치: 아이템 수 + 총 길이 상한 동시 적용
        pending = []
        acc_len = 0
        def flush_batch(batch):
            if not batch:
                return
            results = self._translate_batch_json(batch, fast=True)
            for obj in batch:
                cid = obj["id"]
                original = obj["text"]
                translated = self._strip_meta(results.get(cid) or original)
                key = self._cache_key(original)
                self.cache[key] = translated
            self._save_cache()

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
        j = 0
        for idx, cell in enumerate(cells):
            text = (cell.get_text(" ", strip=True) or "").strip()
            if not text:
                continue
            if not force_cell and len(text) > self.SHORT_CELL_MAX_CHARS:
                continue
            key = self._cache_key(text)
            if key in self.cache:
                cell.string = self.cache[key]

    # -------------------- LLM 호출 유틸 --------------------
    def _translate_batch_json(self, items: List[Dict[str, str]], fast: bool = True) -> Dict[str, str]:
        if not items:
            return {}
        instruction = (
            "You are a translator. Translate each item's 'text' from English to Korean.\n"
            "Respond ONLY with a compact JSON array: [{\"id\": string, \"translated\": string}, ...].\n"
            "Do not add any explanation."
        )
        examples = json.dumps(items, ensure_ascii=False)
        prompt = f"{instruction}\n\nINPUT:\n{examples}\n\nOUTPUT:"
        resp = self._generate_once_fast(prompt) if fast else self._generate_once(prompt)
        try:
            # JSON이 아닌 텍스트가 섞였을 수 있으니 배열 부분만 추출 시도
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
                    "Translate to Korean. Return ONLY the translated text, no explanations:\n"
                    f"{text}\n\nOUTPUT:"
                ) or text
                out[obj["id"]] = self._strip_meta(single)
            return out

    def _generate_once_fast(self, prompt: str) -> Optional[str]:
        """배치 번역용 경량 호출(format=json, 저온도/짧은 예측)."""
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
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                if self.debug_raw_output:
                    self._raw_outputs.append(raw)
                return raw
        except Exception as e:
            logger.error(f"LLM 호출 실패(fast): {e}")
        return None

    def _generate_once(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                **self.quality_settings,
                "temperature": self.temperature,
            },
        }
        try:
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
        return None

    # -------------------- 메타/think 제거 --------------------
    def _strip_meta(self, text: str) -> str:
        if not text:
            return text
        if self.debug_raw_output:
            return text.strip()
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

    # -------------------- 캐시 --------------------
    def _cache_key(self, text: str) -> str:
        norm = re.sub(r"\s+", " ", text.strip())
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()

    def _load_cache(self) -> Dict[str, str]:
        try:
            if self.cache_path.exists():
                return json.loads(self.cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
    
    def _intelligent_text_splitting(self, text: str) -> List[str]:
        """지능적 텍스트 분할"""
        # 주요 구분점으로 분할
        major_splits = re.split(r'(?=^## (?:ARTICLE|APPENDIX|EXHIBIT))', text, flags=re.MULTILINE)
        
        sections = []
        for split in major_splits:
            if not split.strip():
                continue
                
            # 토큰 제한 고려 (약 1200자 = ~800 토큰)
            if len(split) <= 1200:
                sections.append(split.strip())
            else:
                # 긴 섹션은 문단별로 추가 분할
                subsections = self._split_long_section(split)
                sections.extend(subsections)
        
        return sections
    
    def _split_long_section(self, text: str) -> List[str]:
        """긴 섹션 분할"""
        # 문단별 분할 시도
        paragraphs = re.split(r'\n\s*\n', text)
        
        sections = []
        current_section = ""
        
        for para in paragraphs:
            if len(current_section + para) <= 1200:
                current_section += "\n\n" + para if current_section else para
            else:
                if current_section:
                    sections.append(current_section.strip())
                current_section = para
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections
    
    def _final_post_processing(self, text: str) -> str:
        """최종 후처리"""
        # 문서 전체의 용어 일관성 재확인
        text = self._apply_terminology_consistency(text)
        
        # 조항 번호 정리
        text = re.sub(r'(\d+)\.(\d+)\s*([가-힣])', r'\1.\2 \3', text)
        
        # 과도한 공백 정리
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # 문장 부호 정리
        text = re.sub(r'\s+([,.;:])', r'\1', text)
        text = re.sub(r'([,.;:])\s*([가-힣])', r'\1 \2', text)
        
        return text.strip()

# 실행 예시
def main():
    """메인 실행 함수"""
    translator = QualityContractTranslator(
        model_name="gemma3:4b",  # 최고 품질 모델
        temperature=0.1,           # 일관성 우선
        max_retries=3              # 품질 확보를 위한 재시도
    )
    
    # 단일 섹션 테스트
    test_text = """## ARTICLE 5 RIGHTS IN THE GLOBAL SYSTEMS 

5.1 This Agreement does not transfer to Affiliate any rights or potential rights in the Global Systems other than the right to Use as defined in Section 1.7."""
    
    logger.info("단일 섹션 번역 테스트 시작")
    result = translator.translate_with_quality_check(test_text)
    
    print(f"\n원문:\n{result.original_text}")
    print(f"\n번역:\n{result.translated_text}")
    print(f"\n신뢰도: {result.confidence_score:.2f}")
    print(f"처리시간: {result.processing_time:.1f}초")
    
    # 전체 문서 번역 (파일이 있는 경우)
    # report = translator.translate_document_premium('contract.md')
    # print(f"\n번역 리포트: {json.dumps(report, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    main()