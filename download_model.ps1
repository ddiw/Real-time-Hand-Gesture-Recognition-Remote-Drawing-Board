$ErrorActionPreference = "Stop"
$modelDirectory = Join-Path $PSScriptRoot "containers\2-vision-analysis\models"
$modelPath = Join-Path $modelDirectory "hand_landmarker.task"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

New-Item -ItemType Directory -Force $modelDirectory | Out-Null
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
Write-Host "Downloaded MediaPipe Hand Landmarker model to $modelPath"
