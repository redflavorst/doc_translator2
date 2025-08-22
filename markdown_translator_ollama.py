# 사용 예시 스크립트

from quality_contract_translator import QualityContractTranslator
import json
from pathlib import Path

def translate_your_documents():
    """제공해주신 문서들을 번역하는 예시"""
    
    # 고품질 번역기 초기화
    translator = QualityContractTranslator(
        model_name="qwen3:8b",  # 또는 "llama3.1:8b"
        temperature=0.1,           # 낮은 온도로 일관성 확보
        max_retries=3              # 품질을 위한 재시도
    )
    
    # 1. 단일 문서 번역
    print("=== 페이지 1 번역 ===")
    report1 = translator.translate_document_premium(
        input_file='./ppstruct_quick_output/page_0001.md',
        output_file='./ppstruct_quick_output/page_0001_premium_korean.md'
    )
    
    print("=== 페이지 7 번역 ===")
    report2 = translator.translate_document_premium(
        input_file='./ppstruct_quick_output/page_0007.md', 
        output_file='./ppstruct_quick_output/page_0007_premium_korean.md'
    )
    
    # 2. 번역 품질 리포트 출력
    print("\n=== 번역 품질 리포트 ===")
    print(f"페이지 1 평균 신뢰도: {report1['average_confidence']:.2f}")
    print(f"페이지 7 평균 신뢰도: {report2['average_confidence']:.2f}")
    
    if report1['low_quality_sections']:
        print(f"⚠️ 페이지 1 낮은 품질 섹션: {report1['low_quality_sections']}")
    
    if report2['low_quality_sections']:
        print(f"⚠️ 페이지 7 낮은 품질 섹션: {report2['low_quality_sections']}")
    
    # 3. 상세 리포트 저장
    with open('translation_report.json', 'w', encoding='utf-8') as f:
        json.dump({
            'page_0001': report1,
            'page_0007': report2
        }, f, indent=2, ensure_ascii=False)
    
    print("\n번역 완료! 다음 파일들이 생성되었습니다:")
    print("- page_0001_premium_korean.md")
    print("- page_0007_premium_korean.md") 
    print("- translation_report.json")

def translate_all_in_folder(folder: str = './ppstruct_quick_output', pattern: str = 'page_[0-9][0-9][0-9][0-9].md'):
    """폴더 내 페이지별 마크다운을 일괄 번역"""
    translator = QualityContractTranslator(
        model_name="qwen3:8b",
        temperature=0.1,
        max_retries=3
    )
    in_dir = Path(folder)
    out_reports = {}
    for md in sorted(in_dir.glob(pattern)):
        # 산출물 재번역 방지: *_premium_korean.md, *_raw.md, *_clean.md 제외
        if md.stem.endswith(('_premium_korean', '_raw', '_clean')):
            continue
        out_file = md.with_name(md.stem + '_premium_korean' + md.suffix)
        # 이미 번역된 파일이 있으면 건너뜀
        if out_file.exists():
            print(f"skip (exists): {out_file.name}")
            continue
        print(f"번역: {md.name} -> {out_file.name}")
        report = translator.translate_document_premium(str(md), str(out_file))
        out_reports[md.name] = report
    with open(in_dir / 'translation_report.json', 'w', encoding='utf-8') as f:
        json.dump(out_reports, f, indent=2, ensure_ascii=False)
    print("완료")

if __name__ == "__main__":
    
    # 2. 전체 문서 번역
    print("2. 전체 문서 번역")
    translate_all_in_folder()