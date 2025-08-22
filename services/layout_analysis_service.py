# services/layout_analysis_service.py
from pathlib import Path
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import sys
import base64
import traceback

# 테스트에서 모킹 가능한 모듈-레벨 심볼 노출 (지연 임포트 기본값)
paddle = None  # type: ignore
PPStructureV3 = None  # type: ignore
save_structure_res = None  # type: ignore

@dataclass
class LayoutAnalysisResult:
    """레이아웃 분석 결과를 담는 데이터 클래스"""
    success: bool
    pages: List[Dict[str, Any]]
    markdown_files: List[str]
    confidence: float
    processing_time: float
    output_dir: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LayoutAnalysisService:
    """test_test_paddle.py의 로직을 서비스로 래핑"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 기본 설정 (원본과 동일)
        self.use_gpu = self.config.get('use_gpu', True)  # 원본은 True 기본값
        self.det_limit_side_len = self.config.get('det_limit_side_len', 1920)
        self.use_table = self.config.get('use_table', True)
        
        # PP-Structure 초기화
        self._pipeline = None
        
    def analyze_document(self, pdf_path: str, output_dir: str = None, progress_callback=None) -> LayoutAnalysisResult:
        """
        PDF 문서의 레이아웃을 분석하고 마크다운으로 변환
        
        Args:
            pdf_path: 분석할 PDF 파일 경로
            output_dir: 결과를 저장할 디렉토리 (None이면 자동 생성)
            
        Returns:
            LayoutAnalysisResult: 분석 결과
        """
        start_time = time.time()
        
        try:
            # 경로 검증 및 설정 (원본과 동일)
            pdf_path = Path(pdf_path).resolve()
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            if output_dir is None:
                output_dir = Path(f"./output_{pdf_path.stem}_{int(time.time())}")
            else:
                output_dir = Path(output_dir).resolve()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"레이아웃 분석 시작: {pdf_path.name}")
            
            # PP-Structure 파이프라인 초기화
            pipeline = self._get_pipeline()
            
            # 진행 상황 콜백 저장
            if progress_callback:
                self._progress_callback = progress_callback
            else:
                self._progress_callback = None
            
            # 문서 분석 실행 (원본과 동일)
            print(f"예측 시작: {pdf_path.name}")
            result = pipeline.predict(input=str(pdf_path))
            
            # 결과 처리
            processed_result = self._process_analysis_result(result, output_dir, pdf_path)
            
            processing_time = time.time() - start_time
            self.logger.info(f"레이아웃 분석 완료: {processing_time:.2f}초")
            
            return LayoutAnalysisResult(
                success=True,
                pages=processed_result['pages'],
                markdown_files=processed_result['markdown_files'],
                confidence=processed_result['confidence'],
                processing_time=processing_time,
                output_dir=str(output_dir),
                metadata=processed_result.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"레이아웃 분석 실패: {e}")
            return LayoutAnalysisResult(
                success=False,
                pages=[],
                markdown_files=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                output_dir=str(output_dir) if 'output_dir' in locals() else "",
                error=str(e)
            )
    
    def _get_pipeline(self):
        """PP-Structure 파이프라인 초기화 (test_test_paddle.py와 동일)"""
        if self._pipeline is None:
            try:
                # GPU 가용성 확인 (원본과 동일)
                use_gpu = self.use_gpu
                try:
                    import paddle  # type: ignore
                    if not getattr(paddle.device, "is_compiled_with_cuda", lambda: False)():
                        print("GPU 비활성: 설치된 Paddle이 GPU 빌드가 아닙니다. CPU로 전환합니다.")
                        use_gpu = False
                except Exception:
                    print("Paddle 모듈 확인 실패. CPU로 전환합니다.")
                    use_gpu = False

                # PPStructureV3 임포트
                from paddleocr import PPStructureV3  # type: ignore

                # test_layout/test_test_paddle.py와 동일하게 간단한 파라미터로 초기화
                # UVDoc 문서 왜곡 보정 비활성화
                self._pipeline = PPStructureV3(
                    device='cpu',  # CPU 강제 사용
                    use_table_recognition=self.use_table,
                    use_doc_unwarping=False,  # UVDoc 비활성화
                    use_doc_orientation_classify=False  # 문서 방향 분류도 비활성화
                )
                
                print(f"PP-StructureV3 초기화 완료 ({'GPU' if use_gpu else 'CPU'})")
                
            except ImportError as e:
                raise ImportError(f"PaddleOCR 모듈을 찾을 수 없습니다: {e}")
            except Exception as e:
                print("PP-StructureV3 예측 중 오류 발생:")
                print(str(e))
                traceback.print_exc()
                raise RuntimeError(f"PP-StructureV3 초기화 실패: {e}")
                
        return self._pipeline
    
    def _process_analysis_result(self, result, output_dir: Path, pdf_path: Path) -> Dict[str, Any]:
        """
        PP-Structure 결과를 처리하고 마크다운 파일 생성
        (원본 test_test_paddle.py의 결과 처리 로직을 그대로 사용)
        """
        # 결과 JSON 저장 (원본과 동일)
        output_json = output_dir / "result.json"
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"저장: {output_json}")
        
        # 원본의 내장 저장 함수들
        def _try_builtin_save(res_list: List[Any]) -> bool:
            saved = False
            for idx, res in enumerate(res_list):
                if hasattr(res, "save_to_img"):
                    try:
                        out_path = output_dir / f"page_{idx+1:04d}_vis.jpg"
                        res.save_to_img(save_path=str(out_path))
                        print(f"저장(내장시각화): {out_path}")
                        saved = True
                    except Exception:
                        pass
                if hasattr(res, "save_to_json"):
                    try:
                        out_path = output_dir / f"page_{idx+1:04d}_res.json"
                        res.save_to_json(save_path=str(out_path))
                        print(f"저장(내장JSON): {out_path}")
                        saved = True
                    except Exception:
                        pass
            return saved

        def _save_output_images_if_any(page_obj: Dict[str, Any], page_idx: int) -> bool:
            out_imgs = page_obj.get("outputImages") or page_obj.get("output_images")
            if not isinstance(out_imgs, dict):
                return False
            saved_any = False
            for name, b64 in out_imgs.items():
                try:
                    img_bytes = base64.b64decode(b64)
                    out_path = output_dir / f"page_{page_idx+1:04d}_{name}.jpg"
                    out_path.write_bytes(img_bytes)
                    print(f"저장(시각화): {out_path}")
                    saved_any = True
                except Exception:
                    continue
            return saved_any

        # 결과 리스트화 (원본과 동일)
        if isinstance(result, dict) and isinstance(result.get("result"), list):
            result_list = result["result"]
        elif isinstance(result, list):
            result_list = result
        else:
            result_list = []

        markdown_files = []
        
        if result_list:
            # 원본의 시각화 저장 로직
            used_builtin = _try_builtin_save(result_list)
            if not used_builtin:
                any_saved = False
                for i, page_obj in enumerate(result_list):
                    if isinstance(page_obj, dict) and _save_output_images_if_any(page_obj, i):
                        any_saved = True
                if not any_saved:
                    print("시각화 출력이 제공되지 않았습니다. YAML/CLI 기반 시각화 저장을 고려하세요.")

            # 원본의 마크다운 저장 로직
            markdown_files = self._save_markdown_files_original(result_list, output_dir)
            
            # 원본의 폴백 마크다운 생성
            if not markdown_files:
                markdown_files = self._generate_fallback_markdown_original(result_list, output_dir, pdf_path)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(result_list)
        
        return {
            'pages': result_list,
            'markdown_files': markdown_files,
            'confidence': confidence,
            'metadata': {
                'total_pages': len(result_list),
                'output_json': str(output_json)
            }
        }
    
    def _save_markdown_files_original(self, result_list: List, output_dir: Path) -> List[str]:
        """원본의 마크다운 저장 로직"""
        markdown_files = []
        
        # save_structure_res 확인 (원본과 동일)
        _HAS_SAVE = False
        try:
            from paddleocr import save_structure_res  # type: ignore
            _HAS_SAVE = True
        except Exception:
            _HAS_SAVE = False

        md_saved = 0
        page_md_saved = 0
        
        # 원본의 save_structure_res 로직
        if _HAS_SAVE:
            for i, page_res in enumerate(result_list):
                try:
                    # 최신 시그니처
                    save_structure_res(page_res, save_folder=str(output_dir), 
                                     file_name=f"page_{i+1:04d}", output_format="md")
                    md_file = output_dir / f"page_{i+1:04d}.md"
                    if md_file.exists():
                        md_saved += 1
                        markdown_files.append(str(md_file))
                        print(f"저장(MD): {md_file}")
                except TypeError:
                    try:
                        # 구 시그니처
                        save_structure_res(page_res, str(output_dir), f"page_{i+1:04d}")
                        md_file = output_dir / f"page_{i+1:04d}.md"
                        if md_file.exists():
                            md_saved += 1
                            markdown_files.append(str(md_file))
                            print(f"저장(MD): {md_file}")
                    except Exception:
                        pass
                except Exception:
                    pass

        # 원본의 페이지별 markdown 속성 저장 로직
        try:
            total_pages = len(result_list)
            for i, page_res in enumerate(result_list):
                # 진행 상황 콜백 호출
                if hasattr(self, '_progress_callback') and self._progress_callback:
                    self._progress_callback(i + 1, total_pages, f"페이지 {i+1}/{total_pages} 마크다운 변환 중")
                md_info = None
                if hasattr(page_res, 'markdown'):
                    md_info = getattr(page_res, 'markdown', None)
                elif isinstance(page_res, dict) and 'markdown' in page_res:
                    md_info = page_res.get('markdown')
                if isinstance(md_info, dict):
                    md_text = md_info.get('markdown_texts') or md_info.get('markdown') or ''
                    if md_text:
                        page_md_path = output_dir / f"page_{i+1:04d}.md"
                        page_md_path.write_text(md_text, encoding='utf-8')
                        print(f"저장(MD-페이지): {page_md_path}")
                        page_md_saved += 1
                        # 페이지 저장 완료 시 콜백
                        if hasattr(self, '_progress_callback') and self._progress_callback:
                            self._progress_callback(i + 1, total_pages, f"페이지 {i+1} 마크다운 저장 완료")
                        if str(page_md_path) not in markdown_files:
                            markdown_files.append(str(page_md_path))
                    images = md_info.get('markdown_images') or {}
                    if isinstance(images, dict) and images:
                        for relpath, image in images.items():
                            try:
                                file_path = output_dir / relpath
                                file_path.parent.mkdir(parents=True, exist_ok=True)
                                image.save(file_path)
                                print(f"저장(MD-이미지): {file_path}")
                            except Exception:
                                pass
        except Exception:
            pass
            
        return markdown_files
    
    def _generate_fallback_markdown_original(self, result_list: List, output_dir: Path, pdf_path: Path) -> List[str]:
        """원본의 폴백 마크다운 생성 로직"""
        print("내장 마크다운 저장이 실패하여 폴백으로 합본 Markdown을 생성합니다.")
        
        def _page_items(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
            for key in ["layout", "elements", "items", "result", "res", "layout_parsing", "layoutParsingResults"]:
                val = obj.get(key)
                if isinstance(val, list) and val:
                    return [v for v in val if isinstance(v, dict)]
            return [obj] if isinstance(obj, dict) else []

        def _item_text(it: Dict[str, Any]) -> str:
            return it.get("text") or it.get("res", {}).get("text") or it.get("content") or ""

        def _item_bbox(it: Dict[str, Any]):
            bbox = it.get("bbox") or it.get("box") or it.get("rect") or it.get("poly")
            if isinstance(bbox, list) and len(bbox) >= 8:
                xs = bbox[0::2]; ys = bbox[1::2]
                return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(b) for b in bbox]
            return [0.0, 0.0, 0.0, 0.0]

        def _item_type(it: Dict[str, Any]) -> str:
            return str(it.get("type") or it.get("label") or "text").lower()

        # 원본과 동일한 합본 마크다운 생성
        lines: List[str] = [f"# {pdf_path.name}"]
        for i, page in enumerate(result_list):
            lines.append(f"\n## Page {i+1}")
            items = _page_items(page)
            items.sort(key=lambda it: (_item_bbox(it)[1], _item_bbox(it)[0]))
            for it in items:
                typ = _item_type(it)
                text = _item_text(it).strip()
                if not text:
                    continue
                if typ in {"title", "heading", "header"}:
                    lines.append(f"### {text}")
                else:
                    lines.append(text)
        
        md_path = output_dir / f"{pdf_path.stem}.md"
        md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        print(f"저장(MD-합본): {md_path}")
        
        return [str(md_path)]
    
    def _calculate_confidence(self, result_list: List) -> float:
        """분석 결과의 신뢰도 계산"""
        if not result_list:
            return 0.0
        
        total_confidence = 0.0
        item_count = 0
        
        for page in result_list:
            if isinstance(page, dict):
                # 각 페이지의 아이템들에서 신뢰도 추출
                items = []
                for key in ["layout", "elements", "items", "result", "res"]:
                    val = page.get(key)
                    if isinstance(val, list):
                        items.extend(val)
                
                for item in items:
                    if isinstance(item, dict):
                        # 신뢰도 값 찾기
                        conf = item.get("confidence") or item.get("score") or item.get("prob")
                        if conf is not None:
                            try:
                                total_confidence += float(conf)
                                item_count += 1
                            except (ValueError, TypeError):
                                continue
        
        return total_confidence / item_count if item_count > 0 else 0.8  # 기본값