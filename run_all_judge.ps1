<#
.SYNOPSIS
    Judge ALL generation files in results/generations/ using qwen3.5:35b via Ollama.

.DESCRIPTION
    Iterates over every .jsonl file in results/generations/ and runs
    `python run_evals.py judge-two-pass` for each one.
    Output goes to results/judgments/<same_filename>.

    Requires:
      - Ollama running with qwen3.5:35b loaded (1-2x A100)
      - Environment variables OPENAI_API_KEY=ollama, OPENAI_BASE_URL=http://<host>:11434/v1

.PARAMETER OllamaHost
    Ollama server address (default: 127.0.0.1:11434)

.PARAMETER EnableReasoning
    If set, passes --reasoning to enable thinking mode.
    Default: reasoning OFF (--no-reasoning).

.PARAMETER MaxConcurrent
    Concurrent records being judged (default: 4).
    With 35B on 1x A100 keep low (2-4); on 2x A100 can try 4-8.

.PARAMETER RequestTimeout
    Per-request timeout in seconds (default: 300).
    35B is slow on long answers; 300s is safe.

.PARAMETER MaxTokens
    Max tokens for judge response (default: 32).
    Judge should output a single number — 32 tokens is plenty.

.PARAMETER DryRun
    Show commands without executing them.

.EXAMPLE
    # Reasoning OFF (recommended — faster, cleaner output)
    .\run_all_judge.ps1

    # Reasoning ON
    .\run_all_judge.ps1 -EnableReasoning

    # Remote Ollama server on 2x A100, higher concurrency
    .\run_all_judge.ps1 -OllamaHost "192.168.1.100:11434" -MaxConcurrent 8

    # Dry run to see commands
    .\run_all_judge.ps1 -DryRun
#>

param(
    [string]$OllamaHost = "127.0.0.1:11434",
    [switch]$EnableReasoning,
    [int]$MaxConcurrent = 4,
    [float]$RequestTimeout = 300.0,
    [int]$MaxTokens = 32,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ── Configure environment for Ollama ──
$env:OPENAI_API_KEY = "ollama"
$env:OPENAI_BASE_URL = "http://${OllamaHost}/v1"

$JudgeModel = "qwen3.5:35b"
$GenerationsDir = Join-Path (Join-Path $PSScriptRoot "results") "generations"
$JudgmentsDir = Join-Path (Join-Path $PSScriptRoot "results") "judgments"

if (-not (Test-Path $GenerationsDir)) {
    Write-Error "Generations directory not found: $GenerationsDir"
    exit 1
}

New-Item -ItemType Directory -Path $JudgmentsDir -Force | Out-Null

$ReasoningFlag = if ($EnableReasoning) { "--reasoning" } else { "--no-reasoning" }

$files = Get-ChildItem -Path $GenerationsDir -Filter "*.jsonl" | Sort-Object Name
$totalFiles = $files.Count

if ($totalFiles -eq 0) {
    Write-Warning "No .jsonl files found in $GenerationsDir"
    exit 0
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Judge ALL generations with $JudgeModel" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Ollama endpoint : $env:OPENAI_BASE_URL"
Write-Host "  Reasoning       : $(if ($EnableReasoning) {'ON'} else {'OFF'})"
Write-Host "  Max concurrent  : $MaxConcurrent"
Write-Host "  Request timeout : ${RequestTimeout}s"
Write-Host "  Max tokens      : $MaxTokens"
Write-Host "  Files to judge  : $totalFiles"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$fileIndex = 0
foreach ($file in $files) {
    $fileIndex++
    $inputPath = $file.FullName
    $outputPath = Join-Path $JudgmentsDir $file.Name

    Write-Host "[$fileIndex/$totalFiles] $($file.Name)" -ForegroundColor Yellow

    $cmdArgs = @(
        "run_evals.py"
        "judge-two-pass"
        "--input", $inputPath
        "--output", $outputPath
        "--judge-model", $JudgeModel
        "--max-concurrent", $MaxConcurrent
        "--max-requests-per-second", "0"
        "--request-timeout", $RequestTimeout
        "--judge-max-tokens", $MaxTokens
        "--checkpoint-batch-size", "50"
        "--resume"
        $ReasoningFlag
    )

    if ($DryRun) {
        Write-Host "  [DRY RUN] python $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    } else {
        $startTime = Get-Date
        Write-Host "  Started: $($startTime.ToString('HH:mm:ss'))"

        & python @cmdArgs
        $exitCode = $LASTEXITCODE

        $elapsed = (Get-Date) - $startTime
        if ($exitCode -ne 0) {
            Write-Host "  FAILED (exit code $exitCode) after $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
            Write-Host "  Continuing to next file..." -ForegroundColor Red
        } else {
            Write-Host "  Done in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        }
    }
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  All files processed." -ForegroundColor Cyan
Write-Host "  Judgments in: $JudgmentsDir" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
