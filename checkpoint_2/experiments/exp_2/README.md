# exp_2: imbalance fixes, extra heads and the lean shortcut at full scale

This experiment trains ScoNet-MT variants with the full author schedule and compares them under one protocol. The checkpoint and the threshold are chosen on a held-out val split, never on test. It answers:

1. Which way of fixing the 1:1:8 imbalance works at full scale: stratified batches, upsampling, or the logit-adjusted (LA) loss? (phase 1)
2. Does a neutral head (TG-MILNet's BAM) or an ordinal head add anything on top of that? (phase 2)
3. How much is left when the global silhouette lean from self_learning notebook 08 is removed from the input? (phase 2, `G_deskew`)
4. Do the self_learning ideas that gave nothing on mini data also give nothing at full scale? (phase 2b, `H`-`L`; only `L_flip` was run)
5. Does an order-aware temporal model (ordered clips + temporal conv, optional mirror-invariant posture stream and pose PAV) beat the order-free ScoNet-MT? (phase 3)

`A_base` replaces `../exp_1` (val calibration), which was never run on its own.

**Status (2026-11-22):** 8 phase 1-2 runs (seed 0), 11 phase-3 temporal runs, and `B_strat` seeds 1 and 2. The results are in `analysis/` (`runs.md` per run, `STORY.md` for the narrative).

## On this branch (GitHub `main`)

- `opengait_changes/` holds the OpenGait files this experiment adds or changes (same relative paths as in OpenGait commit `4d228e1`): `sampler.py`, `transform.py`, `main.py`, `class_triplet.py`, `logit_adjusted.py`, `sconet_robust.py`, `multidataset_model.py`.
- The scripts expect to run from an OpenGait repo root that contains `checkpoint_2/experiments/exp_2/`. The ready-to-run tree is branch [`exp2-kaggle`](https://github.com/dindawijayanto/scoliosis-gait-reproduction/tree/exp2-kaggle) (OpenGait fork with all changes); `code/kaggle_Bseeds.ipynb` clones it.
- `code/legacy/` keeps the first runners (`runpod_all.sh`, `kaggle_two_runs.ipynb`) from the old self-contained `checkpoint_2/exp_2/` copy (commit `58aba03`); they use that older layout.
- Not on GitHub: `runs/` (logits and logs, about 3 GB), checkpoints (`*.pt`), the dataset.

## Folder layout

| folder | what is in it | in git |
|---|---|---|
| `code/` | every script of the experiment (configs, training queue, logit export, calibration, figures) | yes |
| `configs/` | the 8 OpenGait configs that were actually trained (seed 0) | yes |
| `partitions/` | train/val and train/test split files | yes |
| `analysis/` | results of every run: tables, bootstrap, figures, the results story, Grad-CAM notebook | yes |
| `runs/` | raw output: `logits/` (val/test logits of every checkpoint), `logs/` (train/export logs), `incoming/` (archives downloaded from lab, Kaggle, RunPod) | no (about 1 GB) |

Model checkpoints stay where OpenGait writes them: `output/Scoliosis1K/ScoNetRobust/<run>/checkpoints/` (repo root). Locally only `A_base_s0` and `F_ord_on_B_strat_s0` have their 16k `.pt`.

## Runs that were trained

All runs: author config (SGD 0.1, batch 8 children x 8 clips, 30 frames), 16k iterations with the lr drop at 10k, seed 0, train 476/59/59, val 120/15/15, official test 749.

| run | sampler | loss | head | input | trained on |
|---|---|---|---|---|---|
| `A_base_s0` | TripletSampler (natural 80/10/10) | CE | - | - | lab (RTX 3060) |
| `B_strat_s0` | stratified: every batch ~3/3/2 children per class | CE | - | - | Kaggle batch 1 |
| `D_la_s0` | natural | LA, tau 1 | - | - | RunPod (A40) |
| `E_bam_on_D_la_s0` | natural | LA | BAM: neutral vs rest | - | Kaggle batch 2 |
| `F_ord_on_D_la_s0` | natural | LA | ordinal: y >= neutral, y >= positive | - | RunPod |
| `F_ord_on_B_strat_s0` | stratified | CE | ordinal | - | lab |
| `G_deskew_on_D_la_s0` | natural | LA | - | global lean removed | Kaggle batch 2 |
| `L_flip_on_D_la_s0` | natural | LA | - | random horizontal flip in training | Kaggle batch 1 |
| `B_strat_s1`, `B_strat_s2` | stratified | CE | - | - | Kaggle (`code/kaggle_Bseeds.ipynb`), 2026-11-22 |

Phase 3 (ordered 30-frame clips, stratified sampler, CE), RunPod 3x A40, 2026-11-21:

| run | what | seeds |
|---|---|---|
| `T_hppmax_on_B_strat` | control: fine-tune of `B_strat_s0`@16k for 4k iterations, per-frame HPP then max (order-free) | 0, 1, 2 |
| `T_tconv_on_B_strat` | + temporal conv over the ordered frames, per strip | 0, 1, 2 |
| `T_pose_on_B_strat` | + temporal conv + mirror-invariant posture stream | 0, 1, 2 |
| `S_tconv_strat`, `S_pose_strat` | the same two models trained from scratch (16k) | 0 |

Prepared but not run: `H_strat_la` (stratified + LA on the data prior), `F_ord_on_H_strat_la`, `ALL_strat_tconv_la` (3 seeds).

The configs in `configs/` are the ones from the platform each run was trained on; only `dataset_root` and `num_workers` differ between platforms. The partition path was updated to this folder. `F_ord_on_B_strat_s0.yaml` was not saved by the lab, so it was rebuilt with the same recipe (`make_configs.build`, base `B_strat`, `total_iter` 16000).

Planned but not run: `C_up`, `C_sqrt` (upsampling), `H_tattn`, `I_strip`, `J_ctri`, `K_gei`, and seeds 1 and 2 of everything except `B_strat`. `code/make_configs.py` still generates them.

## How the variants work

- **The LA loss reads its prior from the sampler.** With a stratified sampler the prior is uniform, so LA is the same as CE there. That is why `B_strat` uses plain CE: correcting twice would push the model toward the minority classes.
- **Stratified vs upsampling** differ only in batch composition. Stratified gives ~3/3/2 every batch; upsampling gives 1/3 on average, but 3.8% of the batches still have no neutral (natural sampler: 43.8%, measured on this split).
- **Ordinal head:** negative < neutral < positive. Its first task (y >= neutral) gets gradient from every scoliosis child, not only from the 59 neutrals. That targets the main error: 97% of the missed neutrals go to negative.
- **Deskew:** each frame gets a horizontal shear that makes its best-fit body axis vertical. Local bends stay. On 90 test children it brings the notebook 08 asymmetry (upper torso 3.0 / 4.2 / 4.7, negative / neutral / positive) down to about 0. Caveat: a real trunk lean caused by scoliosis is removed too, so read a drop under `G` as "this much came from the global lean", not as "this much was a shortcut".
- No flip anywhere except `L_flip` (notebook 08 predicted it would hurt).

## Code

Scripts in `code/` find their own folder, so they can be started from any directory.

| file | what |
|---|---|
| `code/make_configs.py` | writes the configs (variants x 3 seeds) into `configs/`; `--base` picks the phase-1 variant that phase 2 builds on |
| `code/setup_lab.sh` | one-time setup on the Linux lab server (venv, data check, configs) |
| `code/run_queue.sh` | trains a phase or a list of runs, `JOBS` runs per GPU, skips finished runs; logs go to `runs/logs/` |
| `code/export_logits.py` | val + test logits of every checkpoint into `runs/logits/<run>/`; at 16k also the mirrored test (lean-shortcut probe), the deskewed test and the 256x16 embeddings. Needs Linux (gloo is broken on the laptop) |
| `code/calibrate.py` | chooses checkpoint / tau / aux weight on val, then reports test: `analysis/summary.{md,json}`, `analysis/curves.json`, AUC on normal / mirrored / deskewed test, 5-NN separability, paired bootstrap |
| `code/make_figs.py` | PPT figures `analysis/figs/*.png` and `analysis/bootstrap.md` (bootstrap vs `A_base`) |
| `code/temporal_report.py` | phase-3 tables `analysis/temporal.md`: every temporal run at val tau, shuffle / mirror / deskew probes, temporal vs control over seeds |
| `code/pav_fusion.py` | DRF-style PAV from the pose keypoints, logistic regression on the train children, late fusion with a run (lambda and tau on val) -> `analysis/pav_fusion_<run>.md` |
| `code/relative_asym.py` | global (camera-sensitive) vs relative (roll-invariant) trunk tilts from the keypoints, recording-block effect, random-roll check -> `analysis/relative_asym.md` |
| `code/kaggle_Bseeds.ipynb`, `code/kaggle_Bseeds.sh` | Kaggle (T4 x2) runner for `B_strat` seeds 1-2 from a git clone of branch `exp2-kaggle` |
| `analysis/gradcam/gradcam.ipynb` | Grad-CAM, class-average heatmaps, band shares, left/right vs lean, deletion and sanity checks |

Library code that the configs use lives in the OpenGait tree, not here:

| file | what |
|---|---|
| `opengait/data/sampler.py` -> `ClassAwareSampler` | stratified / upsample sampler with a `class_prior` property |
| `opengait/modeling/losses/logit_adjusted.py` -> `LogitAdjustedCE` | LA cross entropy per strip; tau=0 is exactly ScoNet's CE |
| `opengait/modeling/losses/class_triplet.py` -> `ClassTripletLoss` | triplet with the class as identity (`J_ctri`) |
| `opengait/modeling/models/sconet_robust.py` -> `ScoNetRobust` | ScoNet-MT plus optional `aux_head: bam / ordinal`, frame attention, strip weights, GEI channel, flip, `temporal: tconv / hpp_max` (`TemporalConv`), `posture_stream` (`PostureStream`), `la_prior: sampler / data` |
| `opengait/data/transform.py` -> `DeskewSilCuttingTransform`, `MirrorSilCuttingTransform`, `ShuffleSilCuttingTransform` | lean removal, and the mirror and frame-shuffle probes used at test |
| `opengait/main.py` | `OPENGAIT_SEED` env override (OpenGait otherwise seeds with the rank, always 0 on one GPU) |

## Reproduce

On the lab server, from the repo root:

```bash
DATA=/path/to/Scoliosis1K-sil-pkl bash checkpoint_2/experiments/exp_2/code/setup_lab.sh
tmux new -s exp2
GPUS=0 JOBS=1 bash checkpoint_2/experiments/exp_2/code/run_queue.sh A_base_s0 B_strat_s0   # or phase1 / phase2
python checkpoint_2/experiments/exp_2/code/export_logits.py
```

Copy `runs/logits/` (and the 16k `.pt` files for Grad-CAM) back to the laptop, then:

```bash
python checkpoint_2/experiments/exp_2/code/calibrate.py --ref D_la   # -> analysis/summary.md
python checkpoint_2/experiments/exp_2/code/make_figs.py              # -> analysis/figs, analysis/bootstrap.md
```

Both scripts were rerun after the move to this folder and reproduce `analysis/` byte for byte.

**Cost:** about 2.7 h per run on an A40, about 12 h on the lab 3060 (220 s per 100 iterations, 10.2 GB, so `JOBS=1`).

## Caveats

- **One seed for most runs.** The mini experiments showed a seed std of 2-4 points, so only gaps whose bootstrap CI excludes 0 are claimed. `B_strat` over 3 seeds is 88.6 +- 1.2 (89.7 / 89.2 / 86.9); its gain over `A_base` (seed 0) shrinks from +2.3 [+0.2, +4.5] to +1.3 [-0.5, +3.2].
- **Small val split:** 120 / 15 / 15 children, so one neutral moves val neutral recall by 6.7 points. The chosen tau and aux weight are noisy.
- **Reduced train set:** 594 children, not 744, so compare variants with each other, not with the paper numbers. TG-MILNet is printed only as a reference.
- **Decision rule:** judge a variant by AUC and the paired-bootstrap CI first. Accuracy on the test set (73% scoliosis) rewards bias toward scoliosis.
