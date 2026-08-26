param(
    [string]$NgrokPath = $env:NGROK_PATH,
    [string]$SessionToken = "hand-board",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $NgrokPath) { $NgrokPath = (Get-Command ngrok -ErrorAction SilentlyContinue).Source }
if (-not $NgrokPath -or -not (Test-Path -LiteralPath $NgrokPath)) {
    throw "ngrok not found. Set NGROK_PATH or pass -NgrokPath."
}

$env:SESSION_TOKEN = $SessionToken
$ngrokLog = Join-Path $projectRoot "ngrok.log"
docker compose -f (Join-Path $projectRoot "docker-compose.yml") up -d --build
$ngrok = Start-Process -FilePath $NgrokPath -ArgumentList @("http", $Port, "--log", $ngrokLog) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru

try {
    $publicUrl = $null
    for ($attempt = 0; $attempt -lt 30 -and -not $publicUrl; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $tunnels = Invoke-RestMethod "http://127.0.0.1:4040/api/tunnels"
            $publicUrl = ($tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1).public_url
        } catch {}
    }
    if (-not $publicUrl) { throw "Could not read ngrok public URL. Check ngrok.log." }
    Write-Host "Monitor: $publicUrl/"
    Write-Host "Phone:   $publicUrl/capture.html?t=$SessionToken"
    Write-Host "Keep this PowerShell window open. Press Enter to stop."
    Read-Host
} finally {
    Stop-Process -Id $ngrok.Id -ErrorAction SilentlyContinue
    docker compose -f (Join-Path $projectRoot "docker-compose.yml") down
}
