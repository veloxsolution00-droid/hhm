# Runs the live evaluation for model configurations A, B, C against the same
# question set. Each config gets a fresh process (empty caches) and unloaded models.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Admin\Downloads\RND2_2\RND2_1\RND"
Set-Location $root
$ids = "A01,A02,B02,D01,F01,H05,I01,J01,K02"
$envs = @(
    @{ name = "cfgA_3b_programmatic"; gen = "qwen2.5:3b"; verify = "false"; vmodel = "qwen2.5:3b" },
    @{ name = "cfgB_3b_3b";           gen = "qwen2.5:3b"; verify = "true";  vmodel = "qwen2.5:3b" },
    @{ name = "cfgC_3b_7b";           gen = "qwen2.5:3b"; verify = "true";  vmodel = "qwen2.5:7b" }
)

function Stop-Backend {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*uvicorn app.main:app*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}

function Unload-Models {
    foreach ($m in @("qwen2.5:3b", "qwen2.5:7b")) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -TimeoutSec 60 `
                -ContentType "application/json" -Body (@{ model = $m; keep_alive = 0 } | ConvertTo-Json) | Out-Null
        } catch { }
    }
    Start-Sleep -Seconds 2
}

function Start-Backend {
    Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $root -WindowStyle Hidden
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
            if ($h.status) { return $h }
        } catch { }
    }
    throw "backend did not become healthy"
}

foreach ($cfg in $envs) {
    Write-Host "=== $($cfg.name) ==="
    Stop-Backend
    Unload-Models
    $env:GENERATION_MODEL = $cfg.gen
    $env:VERIFICATION_MODEL = $cfg.vmodel
    $env:VERIFY_ANSWERS = $cfg.verify
    $h = Start-Backend
    Write-Host "health: status=$($h.status) gen=$($h.ollama.generation_model) ver=$($h.ollama.verification_model)"
    python evaluation\audit\live_eval.py --label $cfg.name --ids $ids --out ("evaluation\audit\live_" + $cfg.name + ".json") 2>&1 |
        Tee-Object -FilePath ("evaluation\audit\live_" + $cfg.name + ".log")
    Stop-Backend
}

# Restore the repository's own .env configuration for subsequent work.
Remove-Item Env:GENERATION_MODEL, Env:VERIFICATION_MODEL, Env:VERIFY_ANSWERS -ErrorAction SilentlyContinue
Write-Host "all configurations finished"
