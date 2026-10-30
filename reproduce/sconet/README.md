# ScoNet from Scratch (Scoliosis1K)

A plain-PyTorch re-implementation of **ScoNet** / **ScoNet-MT** for video-based scoliosis screening from gait silhouettes, written as a single notebook (`sconet_from_scratch.ipynb`). It is checked against the paper *Gait Patterns as Biomarkers: A Video-Based Approach for Classifying Scoliosis* and the official [OpenGait](https://github.com/ShiqiYu/OpenGait) code ([`sconet.py`](https://github.com/ShiqiYu/OpenGait/blob/master/opengait/modeling/models/sconet.py), [`sconet_scoliosis1k.yaml`](https://github.com/ShiqiYu/OpenGait/blob/master/configs/sconet/sconet_scoliosis1k.yaml)).

No OpenGait dependency is needed to run the notebook: the backbone, heads, losses, sampler, training loop and evaluation are all implemented inline.

## Task

Classify each silhouette sequence into one of three classes:

| Class ID | Label |
|---|---|
| 0 | `negative` |
| 1 | `neutral` |
| 2 | `positive` |

## Model variants

Selected with `MODEL_VARIANT` in the settings cell.

| Variant | Objective | Notes |
|---|---|---|
| `ScoNet` | Cross-entropy only | Plain ScoNet as described in the paper text |
| `ScoNet-MT` | Cross-entropy + triplet loss (on subject IDs) | Matches the current official OpenGait YAML |

## Architecture

Input is a clip of 30 silhouettes, each 64 x 44 (the raw 64 x 64 frame is cropped by 10 px on each side).

| Stage | Output shape |
|---|---|
| Input silhouettes | `[N, T, 64, 44]` |
| Frame-wise ResNet9 backbone (channels 64/128/256/512, strides 1/2/2/1, no max-pool) | `[N, 512, T, 16, 11]` |
| Temporal max pooling | `[N, 512, 16, 11]` |
| Horizontal pooling (16 parts, mean + max) | `[N, 512, 16]` |
| SeparateFCs (independent 512 -> 256 per part) | `[N, 256, 16]` (embedding) |
| SeparateBNNecks (BatchNorm + per-part cosine classifier) | `[N, 3, 16]` (logits) |

Total parameters: **7,014,592**.

## Losses and optimisation

- **Cross-entropy:** applied per part, logit scale 16, label smoothing 0.1.
- **Triplet loss** (`ScoNet-MT` only): all-pairs triplet loss on the 256-d embeddings, margin 0.2, using subject IDs as triplet labels.
- **Optimiser:** SGD, lr 0.1, momentum 0.9, weight decay 5e-4, `MultiStepLR` (gamma 0.1).
- **Mixed precision:** enabled automatically on CUDA (matches `enable_float16=true` in the YAML).
- **Batching:** P identities x K sequences per iteration, 30 random frames per sequence.

### Training profiles

Selected with `PROFILE`. LR milestones are scaled to 50% / 70% / 90% of the total iterations (the `paper` profile corresponds to the YAML's 10k / 14k / 18k steps).

| Profile | Iterations | Identities x sequences per batch |
|---|---|---|
| `quick` | 200 | 4 x 4 |
| `scaled` | 2,000 | 8 x 4 |
| `paper` | 20,000 | 8 x 8 |

## Dataset

Uses the **Scoliosis1K** silhouette pickles (`Scoliosis1K-sil-pkl`) and a partition JSON (`Scoliosis1K_118.json` by default, the paper's main 1:1:8 setting).

Expected layout:

```
Scoliosis1K-sil-pkl/
  <subject_id>/
    <negative|neutral|positive>/
      <view>/
        <view>.pkl      # uint8 array [frames, 64, 64], values 0..255
```

The partition JSON must contain `TRAIN_SET` and `TEST_SET` lists of subject IDs. The notebook checks that the two sets do not overlap and prints the class counts next to the paper's reference counts.

Split `118` (as indexed by the notebook):

| | negative | neutral | positive | total |
|---|---|---|---|---|
| Train | 596 | 74 | 74 | 744 |
| Test | 204 | 126 | 419 | 749 |

## Setup

Requirements: Python 3, `torch`, `numpy`, `matplotlib`, Jupyter. A CUDA GPU is strongly recommended for anything beyond the `quick` profile.

```bash
pip install torch numpy matplotlib jupyter
```

The notebook was run with PyTorch 2.6.0 (CUDA 12.4) on an NVIDIA RTX A2000 Laptop GPU (4 GB VRAM).

## How to run

1. Edit `DATASET_ROOT` and `PARTITION_JSON` in section 1 so they point to your local copy of the data (the defaults are Windows paths).
2. Choose `MODEL_VARIANT` and `PROFILE`.
3. Set `ONE_PER_CLASS = False` for a real experiment. When `True`, training uses only 3 sequences (one per class); this is a debugging toy mode, not a reproduction.
4. Optionally set `RUN_SANITY_CHECK = True` for a small end-to-end smoke test.
5. Set `RUN_FULL = True` and run all cells. Section 9 builds the split, trains, evaluates on the full test set and plots the results.

### Other switches

| Setting | Purpose |
|---|---|
| `RESUME` | Resume from a checkpoint (the config tag must match) |
| `SAVE_EVERY` | Checkpoint interval (default 500 iterations) |
| `MONITOR_DURING_TRAINING`, `MONITOR_EVERY`, `MONITOR_PER_CLASS` | Optional evaluation on a fixed class-balanced test subset during training; does not affect gradients |
| `SEED` | Random seed (default 0); cuDNN is set to deterministic mode |

Checkpoints are written to `checkpoints/sconet_<variant>_<split>_<profile>.pt`.

## Evaluation

- Each test sequence uses up to its first **720 frames**.
- Logits are averaged over the 16 parts, then passed through a softmax (scale 16) to get the prediction.
- Reported: confusion matrix, accuracy, macro precision / recall / F1, weighted F1 and per-class precision / recall / F1.
- "Paper-style" sensitivity and specificity are computed as positive-class recall and negative-class recall respectively.

## Recorded result in the notebook

The saved output comes from a **toy run, not a reproduction**: `PROFILE = "quick"` (200 iterations), `MODEL_VARIANT = "ScoNet"`, `ONE_PER_CLASS = True` (3 training sequences), evaluated on the full 749-sequence test set.

Confusion matrix (rows = true, columns = predicted; order: negative, neutral, positive):

|  | pred. negative | pred. neutral | pred. positive |
|---|---|---|---|
| **true negative** | 37 | 43 | 124 |
| **true neutral** | 5 | 33 | 88 |
| **true positive** | 31 | 69 | 319 |

| Metric | Value |
|---|---|
| Accuracy | 51.94% |
| Macro precision / recall / F1 | 44.51% / 40.15% / 39.41% |
| Weighted F1 | 48.94% |
| Recall (negative / neutral / positive) | 0.181 / 0.262 / 0.761 |

The paper reports 51.3% accuracy for ScoNet on split 118 after 20,000 iterations. The similar number here should not be read as a successful reproduction: the model saw only three training sequences and mostly predicts `positive`, which is the majority class of the test set (419 of 749). A proper comparison needs `ONE_PER_CLASS = False` and `PROFILE = "paper"`.

## Notes and caveats

- The paper's text lists 745 training sequences for split 118 but its class counts sum to 744; the notebook indexes 744.
- The paper's sensitivity/specificity reporting does not fully reconcile with its published three-class confusion matrix, so compare those two numbers with care.
- `Scoliosis1K-sil-pkl` and the partition files are not included in this repository; obtain them following the OpenGait / Scoliosis1K instructions.

## References

- Gait Patterns as Biomarkers: A Video-Based Approach for Classifying Scoliosis (ScoNet / ScoNet-MT, Scoliosis1K dataset)
- [OpenGait](https://github.com/ShiqiYu/OpenGait)
