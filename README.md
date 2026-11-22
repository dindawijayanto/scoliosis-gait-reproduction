# Checkpoint 1: reproducing ScoNet (MICCAI 2024)

Paper: *Gait Patterns as Biomarkers: A Video-Based Approach for Classifying Scoliosis* (Zhou et al., MICCAI 2024).
Official code: [OpenGait](https://github.com/ShiqiYu/OpenGait), `opengait/modeling/models/sconet.py` and `configs/sconet/sconet_scoliosis1k.yaml` (commit `4d228e1`).

This folder has two parts:

| Folder | What it is |
|---|---|
| `self_learning/` | My own small notebooks where I rebuilt each piece to understand it: raw → pkl, pkl → ScoNet, and the six Table 4 variants on 12 children. |
| `reproduce/` | The real reproduction: the official OpenGait code and the paper's settings, trained on RunPod, plus the notebooks that analyse the results. |

## What is not on GitHub

- **The Scoliosis1K dataset.** It is released under a signed agreement and is password-protected, so it is in `.gitignore`. Get it from the [project page](https://zhouzi180.github.io/Scoliosis1K/) and see `reproduce/datasets/README.md`.
- **Checkpoints (`*.pt`) and TensorBoard files.** Too big for GitHub. The text logs, the per-checkpoint results and the cached test logits are included, so every notebook still runs and shows its results.

## Checkpoint 2

`checkpoint_2/` asks why ScoNet-MT misses the mild (neutral) class and whether published fixes really help:

| Folder | What it is |
|---|---|
| `checkpoint_2/self_learning/` | Small-data notebooks 01-08: class imbalance, strip patterns, temporal pooling, ScoliGaitX, TG-MILNet, a robust pipeline, the flip / lean asymmetry. |
| `checkpoint_2/experiments/exp_1/` | Val-split calibration of the checkpoint 1 model. |
| `checkpoint_2/experiments/exp_2/` | Full-scale runs: 8 imbalance / head / input variants, temporal models, extra seeds, and the analysis (`analysis/runs.md`, `analysis/STORY.md`). |
