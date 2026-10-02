---
language:
- en
pretty_name: SWE-Master Teacher Rollout 300x3
tags:
- software-engineering
- trajectories
- tool-use
- supervised-fine-tuning
size_categories:
- n<1K
---

# SWE-Master Teacher Rollout 300x3

Private, versioned training artifact for the first SWE-Master teacher-rollout run. It contains both the immutable raw rollout export and the derived Qwen3 SFT data.

## What `300x3` means

- 300 selected software-engineering tasks.
- 3 independent teacher rollouts requested per task.
- 900 raw trajectories in total.
- The teacher run is identified by `teacher-deepseek-flash-official-300x3`.

This is not a dataset of 80,000 examples. The `80k` suffix means an 81,920-token Qwen3 context limit.

## Data flow

```text
300 tasks × 3 rollouts
        │
        ▼
raw/.../batches/batch-0000..0009.jsonl        900 raw trajectories
        │  validate success, tool calls, submit format and ≤100 steps
        ▼
processed/.../clean_success_all.jsonl          776 accepted trajectories
        │  tokenize with Qwen3 tokenizer and require ≤81,920 tokens
        ├── processed/.../qwen3-80k/token_length_rejected.jsonl   1 rejection
        ▼
processed/.../qwen3-80k/clean_success_80k_qwen3.jsonl            775 trajectories
        │  render training prompt/response and loss ranges
        ▼
processed/.../qwen3-80k/train_qwen3_80k.jsonl                    final SFT input
```

The final training file used for the Qwen3-4B RSFT run is:

`processed/teacher-deepseek-flash-official-300x3-rsft/qwen3-80k/train_qwen3_80k.jsonl`

## Directory and file relationships

| Path | Role | Relationship |
|---|---|---|
| `raw/.../manifest.json` | Run manifest | Records selection, configuration, task list, fingerprints, Docker information and run ID. |
| `raw/.../execution.json` | Execution settings | Records rollout concurrency (`max_workers: 4`). |
| `raw/.../pulled_images.json` | Environment inventory | Docker images pulled for the selected tasks. |
| `raw/.../batches/batch-NNNN.json` | Batch index | Ordered list of task IDs assigned to the matching JSONL batch. |
| `raw/.../batches/batch-NNNN.jsonl` | Raw source of truth | Full rollout records; ten batches together contain 900 trajectories. |
| `processed/.../clean_success_all.jsonl` | Validated examples | 776 successful and structurally valid trajectories from 298 tasks. |
| `processed/.../rejected.jsonl` | First-stage audit | One record per rejected raw trajectory with source coordinates and rejection reasons. |
| `processed/.../report.json` | First-stage statistics | Counts, repository distribution, step histogram and rejection summary. |
| `processed/.../qwen3-80k/clean_success_80k_qwen3.jsonl` | Length-audited examples | 775 examples with Qwen3 sequence length and assistant-loss-token statistics. |
| `processed/.../qwen3-80k/token_length_rejected.jsonl` | Length audit | The single example exceeding the 81,920-token limit. |
| `processed/.../qwen3-80k/token_length_report.json` | Length statistics | Tokenizer, threshold, counts and sequence-length histogram. |
| `processed/.../qwen3-80k/train_qwen3_80k.jsonl` | Training-ready data | Adds rendered `prompt`, `response`, `response_ranges` and `prompt_ids_len`; this is the SFT entry point. |

## How a processed row maps back to raw data

Every accepted row contains:

- `source_file`: raw batch filename, for example `batch-0000.jsonl`.
- `source_line`: one-based line number in that batch.
- `exp_name`: rollout identity, including task and rollout number.
- `trajectory_fingerprint`: stable content fingerprint.
- `instance_id`: software-engineering task identity.

Therefore a processed row can be traced exactly to:

```text
raw/teacher-deepseek-flash-official-300x3/batches/<source_file>
                                                        line <source_line>
```

The same identity fields and fingerprint continue through all three processed datasets. They should be used for joins and audits instead of row position.

## Record schemas

### Raw rollout row

The raw JSONL records preserve rollout inputs, agent/environment configuration, full `trajectory_steps`, output patch, test outputs, reward, token statistics and timing information. The raw batches are archival inputs and are not consumed directly by SFT.

### Validated row (`clean_success_all.jsonl`)

| Field | Meaning |
|---|---|
| `docker_image` | Reproducible task environment. |
| `instance_id` | R2E-Gym task identifier. |
| `input` | Ordered chat/tool trajectory in message form. |
| `step_count` | Number of agent steps. |
| `token_usage_total` | Teacher-run token usage. |
| `reward`, `exit_reason` | Rollout result metadata. |
| `exp_name` | Unique experiment/rollout name. |
| `source_file`, `source_line` | Exact raw provenance. |
| `trajectory_fingerprint` | Stable trajectory fingerprint. |

### Qwen3 length-audited row

Adds `qwen3_sequence_length` and `qwen3_assistant_loss_tokens`.

### Final training row (`train_qwen3_80k.jsonl`)

Adds `prompt`, `response`, `response_ranges` and `prompt_ids_len`. These are the rendered training fields consumed by the RSFT pipeline; `input` remains available for inspection and reprocessing.

## Quality-control summary

- Raw: 900 trajectories from 300 tasks.
- First-stage accepted: 776 trajectories from 298 unique tasks.
- First-stage rejected: 124 trajectories. Reasons can overlap, so reason counts do not sum to 124.
- Qwen3 context filtering: 775 kept, 1 rejected for exceeding 81,920 tokens.
- Final SFT set: 775 rows.
- Successful rollouts per retained task: 23 tasks with one, 72 with two, and 203 with three.

The authoritative detailed counts are stored in `report.json` and `token_length_report.json` rather than duplicated into filenames.

## Version and reproducibility

- Dataset revision: `teacher-rollout-300x3-v1`
- Source SWE-Master code commit: `157c845424ce100711bef85ba833ee489efff51a`
- Integrity metadata: `artifacts/teacher-rollout-300x3.json` in the SWE-Master repository
- Visibility: private; rollout contents may include source code, model traces and environment details and should not be made public without a separate review.

Use the immutable tag for training and evaluation instead of an unpinned `main` revision.
