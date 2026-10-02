# Artifact storage

Source code, configuration, tests, and small manifests belong in Git. Rollouts,
JSONL training data, model weights, checkpoints, and runtime logs do not.

The private artifact repositories are:

- Dataset: `JJerry0000/swe-master-teacher-rollout-300x3`
- Model: `JJerry0000/Qwen3-4B-SWE-RSFT-80K`

The authoritative paths, revisions, checksums, and record counts are recorded in
`artifacts/*.json`. Always pin a revision for reproducible runs and verify the
listed SHA-256 before training or evaluation.

On `swe-master`, Hugging Face traffic uses the reverse SSH tunnel from the local
proxy. Establish the tunnel locally:

```bash
ssh -R 1082:127.0.0.1:1082 swe-master
```

Then export the proxy variables on the server for the lifetime of the command:

```bash
export http_proxy=http://127.0.0.1:1082
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
```

Never commit Hugging Face tokens, GitHub credentials, proxy credentials, or
machine-specific absolute paths. Base-model weights remain a local cache and
are referenced by model ID/revision rather than uploaded again.
