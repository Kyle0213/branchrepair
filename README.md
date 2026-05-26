# Bamboogle API Suite

This document describes how to run the full Bamboogle API evaluation suite with one command. The suite uses an OpenAI-compatible API backend, generates fresh round0 histories, runs baseline methods, runs graph-only NLGraph, and writes a merged summary.

## What It Runs

- Dataset: Bamboogle, sampled from the 125-case full set by default.
- Cases file: `data/bamboogle/cases.json`.
- Round0 prompt style: `claim_atomic`.
- Graph profile: `bamboogle_relation_v7_equal_token`.
- Graph mode: graph-only, focus round 2.
- Baselines: `cot_sc`, `som`, and `origin_mad`.
- Sampling: the launcher defaults to `SAMPLE_SIZE=10` and `SAMPLE_SEED=7`. Each sampled run generates fresh API round0 histories instead of reusing an existing round0 file.
- Safety: `--disable_target_check_landing` is always used for graph evaluation. Gold answers are only used by the evaluator for correctness comparison.

## Environment Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install requests datasets transformers
```

Set your API key in the shell. Do not put the key into scripts, logs, or command files.

```bash
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

For DeepSeek-compatible defaults, no other environment variables are required. To use another OpenAI-compatible endpoint, set:

```bash
export API_BASE_URL="https://api.deepseek.com"
export API_BETA_BASE_URL="https://api.deepseek.com/beta"
export API_MODEL="deepseek-chat"
```

## One-Click Start

```bash
bash scripts/launch_bamboogle_api_suite.sh
```

The launcher starts a `nohup` background process and prints the output directory, PID, and log path.

You can override runtime settings:

```bash
OUTPUT_ROOT="results/bamboogle_api_suite_deepseek_$(date +%Y%m%d_%H%M%S)" \
SHARDS=5 \
SAMPLE_SIZE=10 \
SAMPLE_SEED=7 \
MAX_PARALLEL=5 \
MAX_NEW_TOKENS=1536 \
API_MODEL=deepseek-chat \
bash scripts/launch_bamboogle_api_suite.sh
```

Extra arguments are passed through to `scripts/run_bamboogle_api_suite.py`. Examples:

```bash
bash scripts/launch_bamboogle_api_suite.sh --skip_baselines
bash scripts/launch_bamboogle_api_suite.sh --skip_graph
bash scripts/launch_bamboogle_api_suite.sh --no-resume
bash scripts/launch_bamboogle_api_suite.sh --sample_size 0
```

Use `--sample_size 0` or `SAMPLE_SIZE=0` to run all 125 cases.

## Monitor Progress

Use the output root printed by the launcher:

```bash
tail -f results/bamboogle_api_suite_YYYYMMDD_HHMMSS/suite.nohup.out
tail -f results/bamboogle_api_suite_YYYYMMDD_HHMMSS/master.log
```

Check whether the process is still alive:

```bash
cat results/bamboogle_api_suite_YYYYMMDD_HHMMSS/suite.pid
ps -p "$(cat results/bamboogle_api_suite_YYYYMMDD_HHMMSS/suite.pid)"
```

## Outputs

The main files are:

- `summary.json`: compact final metrics and paths.
- `round0.history.jsonl`: merged round0 API histories.
- `baselines_som_sc_mad.json`: SC, SoM, and origin MAD baseline results.
- `graph_merged.json`: merged graph-only NLGraph result.
- `master.log`: command-level start/end records.
- `shard_XX/`: per-shard cases, round0 history, graph result, and logs.

The run is resumable by default. Reuse the same `OUTPUT_ROOT` and rerun the launcher; completed round0 shards, baseline output, and graph shards are skipped when their case counts match the expected sampled size.

## Direct Python Entry

For debugging without `nohup`:

```bash
python scripts/run_bamboogle_api_suite.py \
  --output_root results/bamboogle_api_suite_debug \
  --api_base_url https://api.deepseek.com \
  --api_beta_base_url https://api.deepseek.com/beta \
  --api_model deepseek-chat \
  --model deepseek-chat \
  --sample_size 10 \
  --sample_seed 7 \
  --shards 5 \
  --max_parallel 5 \
  --max_new_tokens 1536
```

This still reads `OPENAI_API_KEY` from the environment.
