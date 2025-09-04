param(
    [string]$ModelRoot = "D:\models\paddleocr",
    [switch]$SkipLayout,                 # 레이아웃 모델 생략
    [switch]$DisableOrientation,         # 문서 방향 분류 모델 비활성화
    [string]$TextRecModel = 'korean_PP-OCRv5_mobile_rec'  # 텍스트 인식 모델명
)

$ErrorActionPreference = "Stop"

# 프로젝트 루트 추정 (이 스크립트가 scripts/ 아래에 있다고 가정)
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Write-Host "ProjectRoot : $ProjectRoot"
Write-Host "ModelRoot   : $ModelRoot"

# .venv 활성화 (있으면)
$venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating venv at: $venvActivate"
    . $venvActivate
} else {
    Write-Host "No .venv found. Using system Python on PATH."
}

# Python 실행 파일 결정 (.venv 우선)
$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) { $pythonExe = "python" }

# 모델 루트 생성 및 환경변수 설정 (세션 한정)
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
$env:PADDLEOCR_HOME = $ModelRoot
Write-Host "PADDLEOCR_HOME set to: $env:PADDLEOCR_HOME"

# paddleocr import 가능 여부 점검
Write-Host "Checking paddleocr availability..."
$pyCheck = @'
try:
    import paddleocr, paddle
    print("OK - paddleocr is importable")
except Exception as e:
    import sys
    print("ERROR - paddleocr not importable:", e)
    sys.exit(1)
'@

$pyCheck | & $pythonExe -

# 프리워밍 실행
$useLayout = -not $SkipLayout.IsPresent
$useOrientation = -not $DisableOrientation.IsPresent

Write-Host "Prewarming models... (Layout=$useLayout, Orientation=$useOrientation, RecModel=$TextRecModel)"

$pyWarm = @"
from paddleocr import LayoutDetection, PPStructureV3

if ${useLayout}:
    print("- Prewarm LayoutDetection: PP-DocLayout_plus-L")
    LayoutDetection(model_name="PP-DocLayout_plus-L", device="cpu")

print("- Prewarm PPStructureV3 submodules (det/rec/table/cls)")
PPStructureV3(
    device='cpu',
    use_table_recognition=True,
    use_doc_unwarping=False,
    use_doc_orientation_classify=${useOrientation},
    use_textline_orientation=False,
    text_recognition_model_name='${TextRecModel}',
)
print("OK - Models cached locally.")
"@

$pyWarm | & $pythonExe -

# 캐시 결과 요약
$whlDir = Join-Path $ModelRoot "whl"
if (Test-Path $whlDir) {
    Write-Host "\nCached model folders under: $whlDir"
    Get-ChildItem -Path $whlDir -Directory -Recurse | ForEach-Object { Write-Host (" - " + $_.FullName) }
} else {
    Write-Warning "No 'whl' directory found under $ModelRoot. Prewarm may have failed."
}

Write-Host "\nDone. Copy '$ModelRoot' to the offline server and set PADDLEOCR_HOME accordingly."
