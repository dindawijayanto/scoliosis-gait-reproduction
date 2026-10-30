# Baseline CNN from Scratch (Scoliosis1K)

A plain-PyTorch **Baseline CNN** for scoliosis classification from gait silhouettes, written as a single notebook (`cnn_baseline.ipynb`). It is the comparison model for the ScoNet reproduction (see `sconet_from_scratch`) and shares its data loading, preprocessing, batch sampling and evaluation code.

## What "baseline" means here

The ScoNet paper (Section 4.3) defines the Baseline CNN as a model trained with **cross-entropy only, without Horizontal Pooling (HPP) and without BNNeck**. The paper does not give the layer-by-layer architecture, so this notebook makes a transparent adaptation:

- Reuse the `ResNet9` backbone configuration from the official OpenGait ScoNet YAML (channels 64/128/256/512, strides 1/2/2/1, no max-pool).
- Replace the ScoNet head (HPP, SeparateFCs, BNNeck) with **spatial global average pooling + one linear classifier**.
- No triplet loss and no label smoothing.

This is not a claim that it is identical to the authors' internal baseline. Note that OpenGait's own `baseline.py` does include HPP and BNNeck, so it is deliberately not used here.

## Architecture

Input is a clip of 30 silhouettes, each 64 x 44 (the raw 64 x 64 frame is cropped by 10 px on each side).

| Stage | Output shape |
|---|---|
| Input silhouettes | `[N, T, 64, 44]` |
| Frame-wise ResNet9 backbone | `[N, 512, T, 16, 11]` |
| Temporal max pooling | `[N, 512, 16, 11]` |
| Spatial global average pooling | `[N, 512]` |
| Linear classifier | `[N, 3]` |

Total parameters: **4.90M**. The model returns `embedding [N, 512, 1]` and `logits [N, 3, 1]` so it fits the same train/eval interface as the ScoNet notebook.

## Training setup

| Item | Value |
|---|---|
| Loss | Cross-entropy only (scale 1, label smoothing 0) |
| Optimiser | SGD, lr 0.1, momentum 0.9, weight decay 5e-4 |
| LR schedule | `MultiStepLR`, gamma 0.1, milestones 10k / 14k / 18k (only those below the total iterations) |
| Mixed precision | On when CUDA is available |
| Batching | P identities x K sequences per iteration, 30 random frames per sequence |
| Seed | 0, with deterministic cuDNN; `set_seed()` is called before each model is built |

### Profiles

Selected with `PROFILE` in section 8.

| Profile | Iterations | Identities x sequences per batch |
|---|---|---|
| `quick` | 200 | 4 x 4 |
| `scaled` | 2,000 | 8 x 4 |
| `test` | 5,000 | 4 x 4 |
| `paper` | 20,000 | 8 x 8 |

During training the notebook trains in chunks (every `TOTAL_ITERS // 4` iterations) and evaluates a fixed class-balanced test subset (30 per class) after each chunk. Checkpoints go to `baseline_cnn_<profile>.pt`.

## Dataset

Uses the **Scoliosis1K** silhouette pickles (`Scoliosis1K-sil-pkl`) and the partition JSON `Scoliosis1K_118.json` (the paper's main 1:1:8 setting).

```
Scoliosis1K-sil-pkl/
  <subject_id>/
    <negative|neutral|positive>/
      <view>/
        <view>.pkl      # uint8 array [frames, 64, 64], values 0..255
```

`DATASET_ROOT` must point at the folder that directly contains the subject folders (`00000`, `00001`, ...).

Split `118` (as indexed by the notebook):

| | negative | neutral | positive | total |
|---|---|---|---|---|
| Train | 596 | 74 | 74 | 744 |
| Test | 204 | 126 | 419 | 749 |

Because negatives are about 80% of the training set, a batch of 4 identities contains no positive sequence in roughly 66% of iterations. Keep this in mind for the smaller profiles.

## Setup

Requirements: Python 3, `torch`, `numpy`, `matplotlib`, Jupyter. A CUDA GPU is recommended.

```bash
pip install torch numpy matplotlib jupyter
```

The notebook was run with PyTorch 2.6.0 (CUDA 12.4) on an NVIDIA RTX A2000 Laptop GPU (4 GB VRAM).

## How to run

1. Edit `DATASET_ROOT` (section 1) and `PARTITION_JSON` (section 8) to match your machine; the defaults are Windows paths.
2. Run sections 1 to 6 to define the data pipeline, model, loss and training loop.
3. Section 7 is an optional sanity check (a few clips per class, a few iterations). It only verifies that the model, loss and prediction path work end-to-end; it is not a result.
4. In section 8, set `PROFILE` and `ONE_PER_CLASS`:
   - `ONE_PER_CLASS = True` trains on only 3 sequences (one per class), a debugging mode.
   - For the paper budget use `PROFILE = "paper"` and `ONE_PER_CLASS = False`.
5. Set `RUN_FULL = True` and run the final cells to train, evaluate on the full test set and plot loss, accuracy curves and the confusion matrix.

## Evaluation

- Each test sequence uses up to its first **720 frames**.
- Reported: confusion matrix, accuracy, sensitivity, specificity, macro precision / recall / F1 (as OpenGait's `evaluate_scoliosis` logs them) and per-class metrics.
- Sensitivity and specificity follow the numbers implied by the paper's Fig. 4, because the paper's text definitions look swapped:
  - **Sensitivity** = share of true negatives predicted negative.
  - **Specificity** = share of true neutral + positive sequences predicted as neutral or positive.
  - Note this differs from the "paper-style" definition used in the ScoNet notebook (positive-class recall / negative-class recall), so do not compare those two columns directly across the notebooks.

Paper reference (Table 3, Baseline CNN): **45.5% accuracy, 99.0% sensitivity, 27.0% specificity**. This is a reference, not a guaranteed target.

## Recorded results in the notebook

The saved outputs are from debugging runs only; **no final test-set metrics were recorded**.

| Run | Outcome |
|---|---|
| Sanity check (3 samples, 20 iterations) | Cross-entropy 23.202 -> 1.254, training accuracy 0.33, check reported `FAIL`; eval path predicted `negative` for all three samples (1/3 correct). Expected for such a tiny run; it confirms the pipeline executes, nothing more. |
| `test` profile, `ONE_PER_CLASS = True` (3 training sequences, 5,000 planned iterations) | Training loss fell quickly (CE about 1.12 at iteration 1, below 0.01 by iteration 500) with batch accuracy 1.00. This is memorisation of 3 sequences. The run was interrupted (`KeyboardInterrupt`) before the final evaluation. |

To obtain a meaningful comparison against the paper's 45.5% and against ScoNet, rerun with `PROFILE = "paper"` and `ONE_PER_CLASS = False`.

## Notes and caveats

- The paper does not fully specify the baseline head, so results can differ from the paper's.
- The dataset and partition files are not included; obtain them following the OpenGait / Scoliosis1K instructions.

## References

- Gait Patterns as Biomarkers: A Video-Based Approach for Classifying Scoliosis (ScoNet / ScoNet-MT, Scoliosis1K dataset)
- [OpenGait](https://github.com/ShiqiYu/OpenGait)
