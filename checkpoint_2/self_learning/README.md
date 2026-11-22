# checkpoint_2 / self_learning

Learning notebooks for the new architecture, on **small data** (laptop, RTX 3050 4 GB). Full training and RunPod pipelines are not here; they live in `../experiments/` (`exp_1` val calibration, `exp_2` full-scale runs).

Read the notebooks in this order:

| folder | question | training |
|---|---|---|
| `01_class_imbalance` | Why does ScoNet-MT almost never say "neutral"? Compare weighted CE, balanced sampler, logit-adjusted loss, post-hoc adjustment | toy + 4 mini runs |
| `02_strip_pattern` | What do the 16 strips do (body area, separability, correlation, bias)? | none (saved logits) |
| `03_learned_strip` | Learned per-class strip weights: on saved logits (CV) and inside the network (static / dynamic) | 3 mini runs |
| `04_temporal_model` | What max pooling over frames does, and attention over frames / view bags | 3 mini runs |
| `05_scoligaitx` | ScoliGaitX paper: critical reading, GEI-subtract, optical flow, Align Gate Fusion | 3 mini runs |
| `06_tgmilnet` | TG-MILNet paper: critical reading, DTW bags, IBTA, text guidance, BAM | 2 mini runs |
| `07_robust_pipeline` | One pipeline from the surviving pieces: LA loss, balanced BAM, class triplet, flip; tau chosen on a mini val set (20/20/20), 3 seeds, AUC, test-clip spread | 15 mini runs (~1 h) |
| `08_flip_asymmetry` | Why flip hurt: mirror test on saved models, left/right lean of the raw silhouettes, lean per subject-id block (class = id block) | 3 mini runs (weights saved) |

- `tools/common.py`: shared code that was already explained in checkpoint_1 (data, ResNet9, ScoNet head, losses, mini training loop, metrics). The new modules are written inside each notebook.
- Mini setup (`MINI` in `tools/common.py`): 40/40/40 test children from `TEST_SET` (the same in every notebook); ResNet9 starts from GaitBase (CASIA-B gait recognition, never saw Scoliosis1K); 500 iterations, Adam (1e-4 backbone, 1e-3 new modules), 16 frames, 9 children x 2 copies per batch.
- Mini models are tested like they are trained: 10 random clips of 16 frames per child, scores averaged (testing with all 300 frames gave 36.7% instead of 65.8%, see notebook 01 step 7).
- Mini results are **1 seed on 120 test children**: one child = 0.83 points, one neutral child = 2.5 points of neutral recall. Only big differences mean anything; the real comparison happens in `../experiments/exp_2` with the val split.
- Every notebook writes its figures to `output/figures/`.
