param(
    [string]$SessionToken = "hand-board",
    [switch]$Ngrok,
    [string]$NgrokPath = $env:NGROK_PATH
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $projectRoot "tmp\local-services"
$processes = @()

if (-not (Test-Path -LiteralPath $python)) {
    throw ".venv가 없습니다. 먼저: python -m venv .venv"
}
New-Item -ItemType Directory -Force $logDir | Out-Null

function Start-LocalService {
    param([string]$Name, [string[]]$Arguments)
    $stdout = Join-Path $logDir "$Name.out.log"
    $stderr = Join-Path $logDir "$Name.err.log"
    $process = Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $script:processes += $process
}

$env:SESSION_TOKEN = $SessionToken
$env:VISION_ANALYSIS_WS_URL = "ws://127.0.0.1:8760/ingest/{session_id}"
$env:WEB_CANVAS_OUTPUT_URL = "ws://127.0.0.1:8000/ws/canvas-output/{session_id}"
$env:CANVAS_WS_URL = "ws://127.0.0.1:8770/commands/{session_id}"
$env:PATTERN_COMMAND_WS_URL = "ws://127.0.0.1:8761/landmarks"
$env:HAND_LANDMARKER_MODEL_PATH = Join-Path $projectRoot "containers\2-vision-analysis\models\hand_landmarker.task"

try {
    $publicUrl = $null
    if ($Ngrok) {
        if (-not $NgrokPath) {
            $command = Get-Command ngrok -ErrorAction SilentlyContinue
            if ($command) { $NgrokPath = $command.Source }
        }
        if (-not $NgrokPath -or -not (Test-Path -LiteralPath $NgrokPath)) {
            throw "ngrok를 찾을 수 없습니다. NGROK_PATH를 설정하거나 -NgrokPath를 지정하세요."
        }
        $ngrokOut = Join-Path $logDir "ngrok.out.log"
        $ngrokErr = Join-Path $logDir "ngrok.err.log"
        $ngrokProcess = Start-Process -FilePath $NgrokPath -ArgumentList @("http", "8000") `
            -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $ngrokOut -RedirectStandardError $ngrokErr
        $processes += $ngrokProcess
        for ($attempt = 0; $attempt -lt 30 -and -not $publicUrl; $attempt++) {
            Start-Sleep -Milliseconds 500
            try {
                $tunnels = Invoke-RestMethod "http://127.0.0.1:4040/api/tunnels"
                $publicUrl = ($tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1).public_url
            } catch {}
        }
        if (-not $publicUrl) { throw "ngrok 공개 주소를 가져오지 못했습니다. $ngrokErr 로그를 확인하세요." }
        $env:PUBLIC_BASE_URL = $publicUrl
    } else {
        $env:PUBLIC_BASE_URL = ""
    }

    Start-LocalService "web" @("-m", "uvicorn", "app:app", "--app-dir", "containers/0-web", "--host", "127.0.0.1", "--port", "8000")
    Start-LocalService "canvas" @("-m", "uvicorn", "app:app", "--app-dir", "containers/1-canvas", "--host", "127.0.0.1", "--port", "8770")
    Start-LocalService "pattern-command" @("containers/3-pattern-command/app.py")
    Start-LocalService "vision-analysis" @("-m", "containers.2-vision-analysis.app.main")

    Start-Sleep -Seconds 3
    Write-Host "Monitor: http://127.0.0.1:8000/?t=$SessionToken"
    if ($publicUrl) {
        Write-Host "Phone:   $publicUrl/capture.html?t=$SessionToken"
    } else {
        Write-Host "Phone:   로컬 확인만 가능 (휴대폰 연결은 -Ngrok 옵션 사용)"
    }
    Write-Host "Health:  http://127.0.0.1:8762/health"
    Write-Host "Logs:    $logDir"
    Write-Host "이 창에서 Enter를 누르면 로컬 서비스가 종료됩니다."
    Read-Host
} finally {
    foreach ($process in $processes) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -ErrorAction SilentlyContinue }
    }
}
