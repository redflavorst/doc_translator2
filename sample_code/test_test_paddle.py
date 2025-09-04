from pathlib import Path
import json
import sys
import base64
import traceback
from typing import Any, Dict, List


def main():
    pdf_path = Path("./이전가격_0804/특수관계자계약서_FY2023_합본.pdf").resolve()
    output_dir = Path("./ppstruct_quick_output2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)

    # PP-StructureV3만 사용 + save_structure_res (MD 저장) 시도
    _HAS_SAVE = False
    try:
        from paddleocr import PPStructureV3, save_structure_res  # type: ignore
        _HAS_SAVE = True
    except Exception as e:
        try:
            from paddleocr import PPStructureV3  # type: ignore
        except Exception as e2:
            
            print(f"에러: {e2}")
            sys.exit(1)

    # 구성 (영어 문서 기준)
    # GPU 사용 가능 여부 자동 판별 (GPU Paddle 설치가 아니면 CPU로 전환)
    use_gpu = True 
    try:
        import paddle  # type: ignore
        if not getattr(paddle.device, "is_compiled_with_cuda", lambda: False)():
            print("GPU 비활성: 설치된 Paddle이 GPU 빌드가 아닙니다. CPU로 전환합니다.")
            use_gpu = False
    except Exception:
        print("Paddle 모듈 확인 실패. CPU로 전환합니다.")
        use_gpu = False
    det_limit_side_len = 1920  # 텍스트 감지 최대 변 길이
    use_table = True  # 표 인식 활성화

    pipeline = PPStructureV3(
        device='cpu',  # CPU 강제 사용
        use_table_recognition=use_table
    )
    print(f"PP-StructureV3 초기화 완료 ({'GPU' if use_gpu else 'CPU'})")

    print(f"예측 시작: {pdf_path.name}")

    # PP-StructureV3 predict (PDF 직접 입력)
    try:
        result = pipeline.predict(input=str(pdf_path))
    except Exception as e:
        print("PP-StructureV3 예측 중 오류 발생:")
        print(str(e))
        traceback.print_exc()
        sys.exit(1)

    # 결과 저장 (직렬화 불가능한 타입 방어적으로 처리)
    output_json = output_dir / "result.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장: {output_json}")

    # 1) 결과 객체가 save_to_img/save_to_json 제공 시 자동 저장
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

    # 2) 결과 딕셔너리에 outputImages가 있으면 저장
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

    # 결과 리스트화
    if isinstance(result, dict) and isinstance(result.get("result"), list):
        result_list = result["result"]
    elif isinstance(result, list):
        result_list = result
    else:
        result_list = []

    if result_list:
        used_builtin = _try_builtin_save(result_list)
        if not used_builtin:
            # outputImages 저장 시도
            any_saved = False
            for i, page_obj in enumerate(result_list):
                if isinstance(page_obj, dict) and _save_output_images_if_any(page_obj, i):
                    any_saved = True
            if not any_saved:
                print("시각화 출력이 제공되지 않았습니다. YAML/CLI 기반 시각화 저장을 고려하세요.")

        # 마크다운 저장 (내장 유틸) + 페이지별 markdown 속성 저장
        md_saved = 0
        page_md_saved = 0
        if _HAS_SAVE:
            for i, page_res in enumerate(result_list):
                try:
                    # 최신 시그니처
                    save_structure_res(page_res, save_folder=str(output_dir), file_name=f"page_{i+1:04d}", output_format="md")
                    md_file = output_dir / f"page_{i+1:04d}.md"
                    if md_file.exists():
                        md_saved += 1
                        print(f"저장(MD): {md_file}")
                except TypeError:
                    try:
                        # 구 시그니처
                        save_structure_res(page_res, str(output_dir), f"page_{i+1:04d}")
                        md_file = output_dir / f"page_{i+1:04d}.md"
                        if md_file.exists():
                            md_saved += 1
                            print(f"저장(MD): {md_file}")
                    except Exception:
                        pass
                except Exception:
                    pass

        # 페이지 결과에 markdown 딕셔너리가 있으면 직접 저장
        try:
            for i, page_res in enumerate(result_list):
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

        # 내장 저장이 실패하면 간단한 폴백 마크다운 생성 (문서 합본)
        if md_saved == 0 and page_md_saved == 0:
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

    # 화면 요약 출력
    try:
        items = result_list if result_list else ([result] if result else [])
        first = items[0] if items else None
        print(f"페이지 수: {len(items)}")
        if isinstance(first, dict):
            keys = list(first.keys())
            print({"keys": keys[:10]})
        else:
            print(type(first))
    except Exception:
        pass


if __name__ == "__main__":
    main()