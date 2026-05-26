#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/zihan/silver/Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013"
cd "$PROJECT_ROOT"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it in your shell; do not write it into scripts or logs." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/bamboogle_api_suite_${STAMP}}"
API_BASE_URL="${API_BASE_URL:-https://api.deepseek.com}"
API_BETA_BASE_URL="${API_BETA_BASE_URL:-https://api.deepseek.com/beta}"
API_MODEL="${API_MODEL:-deepseek-chat}"
MODEL="${MODEL:-$API_MODEL}"
SHARDS="${SHARDS:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
API_TIMEOUT="${API_TIMEOUT:-180}"
API_MAX_RETRIES="${API_MAX_RETRIES:-4}"

mkdir -p "$OUTPUT_ROOT"

LOG_PATH="$OUTPUT_ROOT/suite.nohup.out"
PID_PATH="$OUTPUT_ROOT/suite.pid"
COMMAND_PATH="$OUTPUT_ROOT/launch_command.txt"

CMD=(
  python
  scripts/run_bamboogle_api_suite.py
  --output_root "$OUTPUT_ROOT"
  --api_base_url "$API_BASE_URL"
  --api_beta_base_url "$API_BETA_BASE_URL"
  --api_model "$API_MODEL"
  --model "$MODEL"
  --shards "$SHARDS"
  --max_parallel "$MAX_PARALLEL"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --api_timeout "$API_TIMEOUT"
  --api_max_retries "$API_MAX_RETRIES"
)

printf '%q ' "${CMD[@]}" > "$COMMAND_PATH"
printf '%q ' "$@" >> "$COMMAND_PATH"
printf '\n' >> "$COMMAND_PATH"

nohup "${CMD[@]}" "$@" > "$LOG_PATH" 2>&1 &
PID="$!"
printf '%s\n' "$PID" > "$PID_PATH"

echo "Bamboogle API suite launched."
echo "  output_root: $OUTPUT_ROOT"
echo "  pid: $PID"
echo "  log: $LOG_PATH"
echo "  master log: $OUTPUT_ROOT/master.log"
