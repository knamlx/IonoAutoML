$ErrorActionPreference = "Continue"

$workDir = "D:\IonoAutoML"
$outputDir = "D:\IonoAutoML\data_2024_2025_giro"
$runnerLog = "D:\IonoAutoML\parser_giro_2024_2025.runner.log"
$maxAttempts = 24
$sleepSeconds = 300

Set-Location $workDir

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $runnerLog -Value "[$stamp] attempt $attempt/$maxAttempts started"

    $stdout = "D:\IonoAutoML\parser_giro_2024_2025.attempt_$attempt.stdout.log"
    $stderr = "D:\IonoAutoML\parser_giro_2024_2025.attempt_$attempt.stderr.log"

    python .\collect_hf_data.py `
      --config .\config.toml `
      --start 2024-01-01T00:00:00Z `
      --end 2026-01-01T00:00:00Z `
      --output-dir $outputDir `
      --sources giro `
      1> $stdout `
      2> $stderr

    $exitCode = $LASTEXITCODE
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $runnerLog -Value "[$stamp] attempt $attempt finished with exit code $exitCode"

    if ($exitCode -eq 0) {
        Add-Content -Path $runnerLog -Value "[$stamp] GIRO collection completed"
        exit 0
    }

    if ($attempt -lt $maxAttempts) {
        Add-Content -Path $runnerLog -Value "[$stamp] sleeping $sleepSeconds seconds before retry"
        Start-Sleep -Seconds $sleepSeconds
    }
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Add-Content -Path $runnerLog -Value "[$stamp] GIRO collection failed after $maxAttempts attempts"
exit 1
