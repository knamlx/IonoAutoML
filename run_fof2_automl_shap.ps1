$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$Config = ".\configs\experiments\fof2_fast_ml_24h_automl_shap.json"
$StationFile = ".\configs\station_sets\fof2_fast_ml_24h_all_available.txt"
$OutputDir = ".\artifacts\fof2_fast_ml_24h_automl_shap"
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$LogDir = Join-Path $OutputDir "logs"
$LogPath = Join-Path $LogDir "full_batch_$Timestamp.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Config: $Config"
Write-Host "Station file: $StationFile"
Write-Host "Output dir: $OutputDir"
Write-Host "Log: $LogPath"

& .\.venv\Scripts\python.exe .\run_station_batch.py `
  --config $Config `
  --station-file $StationFile `
  --parallel-stations 1 `
  2>&1 | Tee-Object -FilePath $LogPath
