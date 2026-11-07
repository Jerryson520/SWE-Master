# Qwen3-4B SWE-bench Verified 500 official evaluations

## Reproducibility boundary

Large rollouts and evaluator reports live below `$SWE_MASTER_ARTIFACT_ROOT`; Git contains only code, documentation, and checksum manifests. On the evaluation host the archival root is `/home/ecs-user/SWE-Master-artifacts`. Set the environment variable rather than embedding that machine path in scripts.

Both official runs use SWE-bench 3.0.2, the same 500 tasks, prompt, tool schema, context budget, and sampling core: temperature 0.7, top-p 0.8, top-k 20, presence penalty 0, thinking disabled, `max_model_len=81920`, `context_window=81920`, safety margin 8192, 8192 tokens per turn, and at most 150 turns. Task containers have no network.

| Run | Rollouts | Non-empty patches | Reports | Resolved | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 Base | 500 | 174 | 174 | 45 (9.00%) | 0 |
| reasoning-content Full SFT | 500 | 146 | 146 | 50 (10.00%) | 0 |

The Full SFT data uses `leaf-pretokenized-template-v3/promote_to_content_raw`: `reasoning_content` is promoted into ordinary assistant content without `<think>`, existing content is appended with two newlines, and structured tool calls retain their original object and order. The five-task gain is an aggregate training-treatment result, not an isolated causal estimate. The earlier top-p 0.95 run resolved 49/500 and is retained only as a historical control.

## Commands

Build the fixed 30-case diagnostic panel:

```bash
python R2E-Gym/scripts/build_stop_boundary_diagnostic_set.py \
  --base-glob "$SWE_MASTER_ARTIFACT_ROOT/evaluations/qwen3-4b-instruct-2507-base/results/*-shard[0-2].jsonl" \
  --sft-glob "$SWE_MASTER_ARTIFACT_ROOT/evaluations/qwen3-4b-reasoning-full-sft/results/*-shard[0-5].jsonl" \
  --output-dir "$SWE_MASTER_ARTIFACT_ROOT/diagnostics/stop-boundary/qwen3-4b-stop-boundary-panel"
```

Probe Base and Full SFT with identical rendered prefixes:

```bash
python R2E-Gym/scripts/probe_stop_boundary_logprobs.py \
  --input "$SWE_MASTER_ARTIFACT_ROOT/diagnostics/stop-boundary/qwen3-4b-stop-boundary-panel/diagnostic_cases.jsonl" \
  --output-dir "$SWE_MASTER_ARTIFACT_ROOT/diagnostics/stop-boundary/qwen3-4b-stop-boundary-panel" \
  --tokenizer "$QWEN3_BASE_MODEL"
```

Use `R2E-Gym/scripts/evaluate_official_local_images.py --help` for official scoring against locally prepared, network-disabled images. Exact paths and SHA-256 values are in the corresponding manifests under `artifacts/`.
