[CmdletBinding()]
param(
    [switch]$RunIntegration,
    [switch]$RunCorpusEvaluation,
    [string]$ApiUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Continue"
$script:Failures = 0
$script:Warnings = 0

function Write-Check([string]$Name, [bool]$Passed, [string]$Detail = "") {
    if ($Passed) {
        Write-Host "[PASS] $Name $Detail" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $Name $Detail" -ForegroundColor Red
        $script:Failures++
    }
}
function Write-Warn([string]$Name, [string]$Detail = "") {
    Write-Host "[WARN] $Name $Detail" -ForegroundColor Yellow
    $script:Warnings++
}

Write-Host "`n=== Offline RAG local validation ===" -ForegroundColor Cyan
Write-Host "Project: $((Get-Location).Path)"
Write-Host "This script does not modify your documents or Chroma index."
Write-Host "Corpus evaluation writes a JSON report under evaluation/live_results.`n"

# 1. Confirm it is being run from the repository root.
$required = @("app\main.py", "app\rag\chat.py", "tests\test_integration_smoke.py", "requirements.txt")
$missing = @($required | Where-Object { -not (Test-Path $_) })
Write-Check "Repository root" ($missing.Count -eq 0) $(if ($missing.Count) { "Missing: $($missing -join ', ')" } else { "" })
if ($missing.Count -gt 0) {
    Write-Host "Run this script from the project root (the folder containing app and tests)." -ForegroundColor Yellow
    exit 2
}

# 2. Python and package versions.
$python = Get-Command python -ErrorAction SilentlyContinue
Write-Check "Python command" ($null -ne $python) $(if ($python) { $python.Source } else { "python not found on PATH" })
if (-not $python) { exit 2 }

$pyVersion = & python --version 2>&1
Write-Host "       $pyVersion"
$versionOk = $false
try {
    $v = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    $parts = $v.Split(".")
    $versionOk = ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11)
} catch {}
Write-Check "Python >= 3.11" $versionOk

$imports = @(
    @{ Name = "pytest"; Module = "pytest" },
    @{ Name = "Streamlit"; Module = "streamlit" },
    @{ Name = "ChromaDB"; Module = "chromadb" },
    @{ Name = "HTTPX"; Module = "httpx" },
    @{ Name = "Tesseract Python wrapper"; Module = "pytesseract" }
)
foreach ($item in $imports) {
    & python -c "import $($item.Module)" 2>$null
    Write-Check "$($item.Name) import" ($LASTEXITCODE -eq 0)
}
$streamlitVersion = & python -c "import streamlit; print(streamlit.__version__)" 2>$null
if ($LASTEXITCODE -eq 0) { Write-Host "       Streamlit version: $streamlitVersion" }

# 3. Ollama server and configured models.
$ollamaBase = "http://127.0.0.1:11434"
$generationModel = "qwen2.5:3b"
$embeddingModel = "nomic-embed-text"
try {
    $cfg = & python -c "from app.core.config import settings; print(settings.ollama_base_url); print(settings.generation_model); print(settings.embedding_model)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $cfg.Count -ge 3) {
        $ollamaBase = $cfg[0].Trim()
        $generationModel = $cfg[1].Trim()
        $embeddingModel = $cfg[2].Trim()
    }
} catch {}

$models = @()
try {
    $tags = Invoke-RestMethod -Uri "$($ollamaBase.TrimEnd('/'))/api/tags" -TimeoutSec 5
    $models = @($tags.models | ForEach-Object { $_.name })
    Write-Check "Ollama API reachable" $true "($ollamaBase)"
} catch {
    Write-Check "Ollama API reachable" $false "Start Ollama and check port 11434. $($_.Exception.Message)"
}
if ($models.Count -gt 0) {
    $genFound = @($models | Where-Object { $_ -eq $generationModel -or $_.Split(":")[0] -eq $generationModel.Split(":")[0] }).Count -gt 0
    $embFound = @($models | Where-Object { $_ -eq $embeddingModel -or $_.Split(":")[0] -eq $embeddingModel.Split(":")[0] }).Count -gt 0
    Write-Check "Generation model: $generationModel" $genFound
    Write-Check "Embedding model: $embeddingModel" $embFound
    Write-Host "       Installed models: $($models -join ', ')"
} else {
    Write-Warn "Model checks" "Skipped because Ollama is unavailable."
}

# 4. Tesseract executable and OCR wrapper.
$tesseractCandidates = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
)
$tesseractCmd = Get-Command tesseract -ErrorAction SilentlyContinue
if ($tesseractCmd) { $tesseractCandidates += $tesseractCmd.Source }
$tesseractPath = $tesseractCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($tesseractPath) {
    $tessOutput = & $tesseractPath --version 2>&1 | Select-Object -First 1
    Write-Check "Tesseract executable" $true "($tesseractPath; $tessOutput)"
} else {
    Write-Warn "Tesseract executable" "Not found. OCR of scanned PDFs/images will not work until installed or configured. Text PDFs can still work."
}

# 5. Unit/regression suite.
Write-Host "`n=== Unit and regression tests ===" -ForegroundColor Cyan
& python -m pytest -q
$unitExit = $LASTEXITCODE
Write-Check "python -m pytest -q" ($unitExit -eq 0) "(exit code $unitExit)"

# 6. Optional real Ollama + isolated Chroma integration test.
if ($RunIntegration) {
    Write-Host "`n=== Live Ollama + Chroma integration ===" -ForegroundColor Cyan
    $oldIntegration = $env:RND_INTEGRATION
    $env:RND_INTEGRATION = "1"
    & python -m pytest tests\test_integration_smoke.py -q -rs
    $integrationExit = $LASTEXITCODE
    if ($null -eq $oldIntegration) { Remove-Item Env:RND_INTEGRATION -ErrorAction SilentlyContinue }
    else { $env:RND_INTEGRATION = $oldIntegration }
    Write-Check "Live Ollama + Chroma test" ($integrationExit -eq 0) "(exit code $integrationExit)"
} else {
    Write-Warn "Live integration not run" "Rerun with -RunIntegration after confirming both Ollama models are installed."
}

# 7. Check backend health and optionally evaluate the real indexed corpus.
$health = $null
try {
    $health = Invoke-RestMethod -Uri "$($ApiUrl.TrimEnd('/'))/health" -TimeoutSec 8
    Write-Check "FastAPI /health reachable" $true "($ApiUrl)"
    Write-Host "       Overall status: $($health.status)"
    Write-Host "       Ollama available: $($health.ollama.available); models ready: $($health.models_ready)"
    Write-Host "       Chroma available: $($health.chromadb.available); indexed chunks: $($health.chromadb.count)"
    Write-Host "       Metadata available: $($health.metadata.available); documents: $($health.documents)"
    if ($health.status -ne "ok") {
        Write-Warn "Backend reports degraded health" "Inspect the fields above and backend logs before trusting the corpus evaluation."
    }
} catch {
    Write-Warn "FastAPI not running" "Start it with 'python run.py' in another terminal if you want API/corpus evaluation."
}

if ($RunCorpusEvaluation) {
    if ($null -eq $health) {
        Write-Check "Real-corpus evaluation prerequisites" $false "Backend /health is not reachable."
    } else {
        Write-Host "`n=== Real uploaded-corpus evaluation (quick matrix) ===" -ForegroundColor Cyan
        $env:RND2_EVAL_API = $ApiUrl
        & python tests\eval_live_corpus.py --quick
        $evalExit = $LASTEXITCODE
        Remove-Item Env:RND2_EVAL_API -ErrorAction SilentlyContinue
        Write-Check "Real-corpus quick evaluation" ($evalExit -eq 0) "(exit code $evalExit)"
        Write-Host "       Inspect the newest JSON report in evaluation\live_results."
    }
} else {
    Write-Warn "Real-corpus evaluation not run" "Rerun with -RunCorpusEvaluation while FastAPI is running and your documents are indexed."
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Failures: $script:Failures"
Write-Host "Warnings: $script:Warnings"
if ($script:Failures -gt 0) {
    Write-Host "RESULT: FAILED. Fix failed checks before relying on the application." -ForegroundColor Red
    exit 1
}
Write-Host "RESULT: BASIC CHECKS PASSED. Warnings indicate optional checks not run or OCR not installed." -ForegroundColor Green
exit 0
