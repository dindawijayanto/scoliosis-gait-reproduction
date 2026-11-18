# exp_2: imbalance fixes and robust heads at full scale (lab GPU)

This experiment answers three questions with full 20k-iteration training, 3 seeds per variant, checkpoint and threshold chosen on a held-out val split (never on test):

1. Which way of fixing the 1:1:8 imbalance works at full scale: stratified batches, upsampling, or the logit-adjusted (LA) loss? (phase 1)
2. Does a neutral head (TG-MILNet's BAM) or an ordinal head add anything on top of that? (phase 2)
3. How much is left when the global silhouette lean from notebook 08 is removed from the input? (phase 2, `G_deskew`)
4. Do the checkpoint_2 notebook ideas that gave nothing on mini data (frame attention, strip weights, class triplet, GEI-subtract, flip) also give nothing at full scale? (phase 2, `H`-`L`). Optical flow is left out: on 64 x 44 silhouettes it was mostly interpolation (notebook 05), and computing it for 1493 x 300 frames costs more than one run.

It extends `exp_1/val_calibration`, which was never run: variant `A_base` *is* the exp_1 run (author config, reduced train split), so exp_1 does not need its own run any more.

## Variants

All variants use the same author config (SGD 0.1, 20k iterations, batch 8 children x 8 copies, 30 frames). The table lists only what changes.

| variant | sampler | loss | head | input |
|---|---|---|---|---|
| `A_base` | TripletSampler (natural 80/10/10) | CE | - | - |
| `B_strat` | stratified: every batch ~3/3/2 children per class, never a batch without neutral | CE | - | - |
| `C_up` | upsampling, power 1: every class equally likely, random batch mix | CE | - | - |
| `C_sqrt` | upsampling, power 0.5 (prior 59/21/21) | LA, tau 1 | - | - |
| `D_la` | natural | LA, tau 1 | - | - |
| `E_bam_on_<base>` | as `<base>` | as `<base>` | BAM: neutral vs rest | - |
| `F_ord_on_<base>` | as `<base>` | as `<base>` | ordinal: y >= neutral, y >= positive | - |
| `G_deskew_on_<base>` | as `<base>` | as `<base>` | - | global lean removed |
| `H_tattn_on_<base>` | as `<base>` | as `<base>` | gated frame attention instead of max over frames (nb 04) | - |
| `I_strip_on_<base>` | as `<base>` | + CE on the weighted decision | learned per-child strip weights (nb 03) | - |
| `J_ctri_on_<base>` | as `<base>` | + class triplet (nb 07) | - | - |
| `K_gei_on_<base>` | as `<base>` | as `<base>` | - | 2nd channel = frame - sequence mean (ScoliGaitX GEI-subtract, nb 05) |
| `L_flip_on_<base>` | as `<base>` | as `<base>` | - | random horizontal flip in training (nb 07/08, expected to hurt) |

`<base>` is `D_la` by default; after phase 1 you can regenerate the configs on the winner with `--base`.

Notes:
- **The LA loss reads its prior from the sampler.** With a stratified sampler the prior is uniform, so LA is the same as CE there. That is why `B_strat` and `C_up` use plain CE: correcting twice would push the model toward the minority classes.
- **B vs C** differ only in batch composition. Stratified gives exactly ~3/3/2 every batch; upsampling gives 1/3 on average, but 3.8% of the batches still have no neutral (natural sampler: 43.8%, measured on this split).
- **Ordinal head:** negative < neutral < positive. Its first task (y >= neutral) gets gradient from every scoliosis child, not only from the 59 neutrals. That targets the main error: 97% of the missed neutrals go to negative.
- **Deskew:** each frame gets a horizontal shear that makes its best-fit body axis vertical. Local bends stay. On 90 test children it brings the notebook 08 asymmetry (upper torso 3.0 / 4.2 / 4.7, negative / neutral / positive) down to about 0. Caveat: a real trunk lean caused by scoliosis is removed too, so read a drop under `G` as "this much came from the global lean", not as "this much was a shortcut".
- No flip anywhere (notebook 08).

## Code

| file | what |
|---|---|
| `opengait/data/sampler.py` -> `ClassAwareSampler` | stratified / upsample sampler with a `class_prior` property |
| `opengait/modeling/losses/logit_adjusted.py` -> `LogitAdjustedCE` | LA cross entropy per strip; tau=0 is exactly ScoNet's CE (checked) |
| `opengait/modeling/models/sconet_robust.py` -> `ScoNetRobust` | ScoNet-MT plus an optional `aux_head: bam / ordinal`; exports `aux` logits at test |
| `opengait/data/transform.py` -> `DeskewSilCuttingTransform` | lean removal + the usual 10-column cut |
| `opengait/main.py` | `OPENGAIT_SEED` env override (OpenGait otherwise seeds with the rank, always 0 on one GPU) |
| `exp_2/make_configs.py` | writes the 24 configs (8 variants x 3 seeds) |
| `exp_2/run_queue.sh` | runs a phase, `JOBS` runs per GPU at the same time, skips finished runs |
| `exp_2/export_logits.py` | val + test logits (and aux logits) of every checkpoint |
| `exp_2/calibrate.py` | val-chosen checkpoint / tau / aux weight -> test tables (mean +- std over seeds), AUCs, paired bootstrap vs `A_base` |
| `exp_2/gradcam/gradcam.ipynb` | Grad-CAM, class-average heatmaps, band shares, left/right vs lean, frame use, deletion and sanity checks |

## Steps on the lab server

```bash
# 1. copy this repo (with the changes above) and the data to the server, then from the repo root:
DATA=/path/to/Scoliosis1K-sil-pkl bash exp_2/setup_lab.sh
# 2. phase 1 (15 runs) inside tmux
tmux new -s exp2
GPUS=0 JOBS=3 bash exp_2/run_queue.sh phase1   # GPUS = a free GPU from nvidia-smi
# 3. results
python exp_2/export_logits.py
python exp_2/calibrate.py            # -> exp_2/results/summary.md
# 4. phase 2 on the phase-1 winner (default D_la)
python exp_2/make_configs.py --data /path/to/Scoliosis1K-sil-pkl --base D_la
GPUS=0 JOBS=3 bash exp_2/run_queue.sh phase2
python exp_2/export_logits.py && python exp_2/calibrate.py
```

Then copy `output/Scoliosis1K/ScoNetRobust/*/checkpoints/*-16000.pt` (about 56 MB each) and `exp_2/logits/` back to the laptop, and run `exp_2/gradcam/gradcam.ipynb`. It picks up every run it finds.

**Cost:** one run took about 2.7 h on RunPod (16 min per 2000 iterations). Phase 1 is 15 runs and phase 2 is 9, so about 65 GPU-hours. With 3 runs in parallel on one large GPU, that is about a day for phase 1. If time is short, run seed 0 of every variant first (`bash exp_2/run_queue.sh A_base_s0 B_strat_s0 ...`).

## Caveats

- **Small val split:** 120 / 15 / 15 children, so one neutral moves val neutral recall by 6.7 points. The chosen tau and aux weight are noisy; `calibrate.py` prints them per seed.
- **Reduced train set:** training uses 594 children, not 744, so compare variants with each other, not with the paper numbers. TG-MILNet is printed only as a reference.
- **Decision rule:** judge a variant by AUC and the paired-bootstrap CI first. Accuracy on the 1:1:8 test set rewards bias toward scoliosis.
