#!/usr/bin/env bash
#
# Judge ALL generation files in results/generations/ using qwen3.5:35b via Ollama.
#
# Usage:
#   # Reasoning OFF (default, recommended):
#   ./run_all_judge.sh
#
#   # Reasoning ON:
#   ./run_all_judge.sh --reasoning
#
#   # Remote Ollama, higher concurrency:
#   ./run_all_judge.sh --host 192.168.1.100:11434 --concurrent 8
#
#   # Dry run:
#   ./run_all_judge.sh --dry-run
#
#   # All options:
#   ./run_all_judge.sh --host HOST:PORT --reasoning --concurrent N --timeout S --max-tokens N --dry-run

set -euo pipefail

# ── Defaults ──
OLLAMA_HOST="127.0.0.1:11434"
ENABLE_REASONING=false
MAX_CONCURRENT=4
REQUEST_TIMEOUT=300
MAX_TOKENS=32
DRY_RUN=false
JUDGE_MODEL="qwen3.5:35b"

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)         OLLAMA_HOST="$2";      shift 2 ;;
        --reasoning)    ENABLE_REASONING=true;  shift   ;;
        --no-reasoning) ENABLE_REASONING=false; shift   ;;
        --concurrent)   MAX_CONCURRENT="$2";    shift 2 ;;
        --timeout)      REQUEST_TIMEOUT="$2";   shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2";        shift 2 ;;
        --dry-run)      DRY_RUN=true;           shift   ;;
        -h|--help)
            sed -n '2,19p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Configure environment for Ollama ──
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://${OLLAMA_HOST}/v1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATIONS_DIR="${SCRIPT_DIR}/results/generations"
JUDGMENTS_DIR="${SCRIPT_DIR}/results/judgments"

if [[ ! -d "$GENERATIONS_DIR" ]]; then
    echo "Error: Generations directory not found: $GENERATIONS_DIR" >&2
    exit 1
fi

mkdir -p "$JUDGMENTS_DIR"

if $ENABLE_REASONING; then
    REASONING_FLAG="--reasoning"
else
    REASONING_FLAG="--no-reasoning"
fi

# Collect .jsonl files sorted by name
mapfile -t files < <(find "$GENERATIONS_DIR" -maxdepth 1 -name '*.jsonl' -type f | sort)
total_files=${#files[@]}

if [[ $total_files -eq 0 ]]; then
    echo "Warning: No .jsonl files found in $GENERATIONS_DIR"
    exit 0
fi

echo "============================================"
echo "  Judge ALL generations with $JUDGE_MODEL"
echo "============================================"
echo "  Ollama endpoint : $OPENAI_BASE_URL"
echo "  Reasoning       : $(if $ENABLE_REASONING; then echo ON; else echo OFF; fi)"
echo "  Max concurrent  : $MAX_CONCURRENT"
echo "  Request timeout : ${REQUEST_TIMEOUT}s"
echo "  Max tokens      : $MAX_TOKENS"
echo "  Files to judge  : $total_files"
echo "============================================"
echo ""

file_index=0
for input_path in "${files[@]}"; do
    file_index=$((file_index + 1))
    filename="$(basename "$input_path")"
    output_path="${JUDGMENTS_DIR}/${filename}"

    echo "[$file_index/$total_files] $filename"

    cmd=(
        python run_evals.py judge-two-pass
        --input "$input_path"
        --output "$output_path"
        --judge-model "$JUDGE_MODEL"
        --max-concurrent "$MAX_CONCURRENT"
        --max-requests-per-second 0
        --request-timeout "$REQUEST_TIMEOUT"
        --judge-max-tokens "$MAX_TOKENS"
        --checkpoint-batch-size 50
        --resume
        "$REASONING_FLAG"
    )

    if $DRY_RUN; then
        echo "  [DRY RUN] ${cmd[*]}"
    else
        start_time=$(date +%s)
        echo "  Started: $(date +%H:%M:%S)"

        if "${cmd[@]}"; then
            elapsed=$(( $(date +%s) - start_time ))
            printf "  Done in %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
        else
            exit_code=$?
            elapsed=$(( $(date +%s) - start_time ))
            printf "  FAILED (exit code %d) after %02d:%02d:%02d\n" "$exit_code" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
            echo "  Continuing to next file..."
        fi
    fi
    echo ""
done

echo "============================================"
echo "  All files processed."
echo "  Judgments in: $JUDGMENTS_DIR"
echo "============================================"
