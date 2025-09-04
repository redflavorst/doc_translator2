param(
    [string]$ProjectRoot = (Resolve-Path ".").Path,
    [string]$ModelRoot = "D:\models\paddleocr",
    [string]$Wheelhouse = "$ProjectRoot\wheelhouse",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

Write-Host "[1/7] Project root : $ProjectRoot"
Write-Host "[2/7] Model root   : $ModelRoot"
Write-Host "[3/7] Wheelhouse   : $Wheelhouse"

# Ensure paths
New-Item -ItemType Directory -Force -Path $ProjectRoot | Out-Null
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null

# Change to project root
Set-Location $ProjectRoot

# Create venv if missing
if (-not (Test-Path "$ProjectRoot\.venv")) {
    Write-Host "[4/7] Creating virtual environment (.venv)..."
    python -m venv .venv
}

# Activate venv for current session
Write-Host "[4/7] Activating virtual environment..."
. "$ProjectRoot\.venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "[4/7] Upgrading pip..."
python -m pip install --upgrade pip

# Download wheels for offline server
Write-Host "[5/7] Downloading wheels to $Wheelhouse ..."
if (-not (Test-Path "$ProjectRoot\requirements.txt")) {
    throw "requirements.txt not found at $ProjectRoot"
}

pip download -r "$ProjectRoot\requirements.txt" -d "$Wheelhouse"

# Optionally install into this venv (needed to prewarm models)
if (-not $SkipInstall) {
    Write-Host "[5/7] Installing packages into .venv (for prewarm run)..."
    pip install -r "$ProjectRoot\requirements.txt"
}
else {
    Write-Host "[5/7] Skipping install as requested (SkipInstall). Ensure paddleocr is importable before prewarm."
}

# Set PaddleOCR cache location for this session
Write-Host "[6/7] Setting PADDLEOCR_HOME to $ModelRoot"
$env:PADDLEOCR_HOME = $ModelRoot

# Prewarm models (download all needed to local cache)
Write-Host "[6/7] Prewarming PaddleOCR models to $ModelRoot ..."
$py = @'
from paddleocr import LayoutDetection, PPStructureV3
print("- Prewarming: LayoutDetection(PP-DocLayout_plus-L) ...")
LayoutDetection(model_name="PP-DocLayout_plus-L", device="cpu")
print("- Prewarming: PPStructureV3 submodules (det/rec/table/cls) ...")
PPStructureV3(
    device='cpu',
    use_table_recognition=True,
    use_doc_unwarping=False,
    use_doc_orientation_classify=True,
    use_textline_orientation=False,
    text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
)
print("OK - Models cached locally.")
'@

python - << $py

# Quick listing of cached model folders
Write-Host "[7/7] Cached folders under $ModelRoot\whl:"
if (Test-Path "$ModelRoot\whl") {
    Get-ChildItem -Path "$ModelRoot\whl" -Directory -Recurse -Depth 1 | Select-Object FullName
} else {
    Write-Warning "No 'whl' directory found under $ModelRoot. Prewarm may have failed."
}

Write-Host "\nDone. Next steps on offline server:"
Write-Host "1) Copy '$Wheelhouse' and '$ModelRoot' to the server"
Write-Host "2) On server: python -m venv .venv; .\.venv\Scripts\Activate.ps1"
Write-Host "3) pip install --no-index --find-links .\wheelhouse -r requirements.txt"
Write-Host "4) setx PADDLEOCR_HOME '$ModelRoot'  (or your chosen path)"
Write-Host "5) Run: python .\run_server.py"
