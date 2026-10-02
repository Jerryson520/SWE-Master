# SWE-Master evaluation viewer

This directory builds a static browser for paired SFT/RL JSONL runs. The index
loads only compact metadata; complete trajectories are split by task and loaded
only after a row is selected.

From `R2E-Gym`:

```bash
python evaluation_viewer/build_report.py \
  --sft results/50-tasks/swe-master-4b-50tasks-128k-150step.jsonl \
  --rl results/50-tasks-rl/swe-master-4b-rl-50tasks-128k-150step.jsonl \
  --output results/evaluation-viewer

cd results/evaluation-viewer
python -m http.server 8080 --bind 127.0.0.1
```

Open `http://127.0.0.1:8080`. If the server is remote, forward it with
`ssh -L 8080:127.0.0.1:8080 swe-master` first.

The source JSONL files are never modified. Re-running the builder refreshes the
generated report.
