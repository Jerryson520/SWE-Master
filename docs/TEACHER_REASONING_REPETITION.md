# Teacher reasoning repetition audit

`R2E-Gym/scripts/analyze_teacher_reasoning_repetition.py` audits the exact
reasoning-content Full SFT population without changing the training data.

The script joins two kinds of input by `(repo, instance_id, seed,
trajectory_fingerprint)`:

- `filtered/{train,validation}.jsonl` supplies the original OpenAI messages;
- `tokenized/{train,validation}.jsonl` supplies the identities that actually
  passed the 81,920-token gate.

The default accepted-count gate is 694. The process fails closed if an accepted
identity is missing, duplicated, or cannot be joined. The older 775-example
RSFT-80K dataset cannot be substituted for these files.

The audit reports token n-gram self-repetition, cross-turn similarity, overlap
with preceding tool results, literal `<tool_call>` text, and code-like quoting.
All labels are review candidates; none authorizes automatic deletion. In
particular, tool-output overlap that is predominantly code-shaped is also
tagged `legitimate_code_reference` and is not treated as high risk by itself.

Example:

```bash
python R2E-Gym/scripts/analyze_teacher_reasoning_repetition.py \
  --filtered-train /path/to/filtered/train.jsonl \
  --filtered-validation /path/to/filtered/validation.jsonl \
  --tokenized-train /path/to/tokenized/train.jsonl \
  --tokenized-validation /path/to/tokenized/validation.jsonl \
  --tokenizer /path/to/Qwen3-4B-Instruct-2507-tokenizer \
  --output-dir /home/ecs-user/SWE-Master-artifacts/diagnostics/teacher-reasoning-repetition
```

The output includes trajectory- and turn-level JSONL, a ranked CSV, a
deterministic 60-row manual-review sheet, a summary, and a self-contained HTML
report. Raw reasoning excerpts are kept outside Git because they may contain
source code or other private trajectory content.
