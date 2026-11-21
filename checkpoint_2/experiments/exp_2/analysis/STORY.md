# Results story: from reproduction to exp_2 (for PPT and write-up)

Working title: **Imbalance Handling and Shortcut Reliance in Gait-Based Scoliosis Screening: An Empirical Study on Scoliosis1K**

All numbers below come from the merged runs (status 2026-11-19, seed 0 only). Tables: `analysis/summary.md`, bootstrap: `analysis/bootstrap.md`, figures: `analysis/figs/`, Grad-CAM: `analysis/gradcam/output/figures/`, one section per training run: `analysis/runs.md`.

## 0. The story in one paragraph

ScoNet-MT reproduces (82.6 acc), but it almost never recognises the mild (neutral) class: neutral recall is 40.5 and 97% of the misses go to negative. The model can rank neutrals correctly (AUC neutral-vs-negative 0.95), so the failure is the decision border, not the features. When every model gets a threshold chosen on a held-out validation split, most published-style imbalance fixes (logit-adjusted loss, BAM head, ordinal head) lose their advantage. Only the class-stratified sampler still gives a significant gain (bal 89.7, +2.3 over ScoNet-MT). Underneath, all models share a shortcut: they read the global body lean. Mirroring the input inverts them (AUC 97 -> 27). Training on deskewed input halves this dependence (mirror AUC 59) at a cost of about 2 points.

## 1. Where each run came from

| variant | what changes | trained on | 16k checkpoint |
|---|---|---|---|
| A_base | ScoNet-MT, natural sampler, CE | lab KC (RTX 3060) | yes |
| B_strat | class-stratified sampler (every batch has all 3 classes) | Kaggle batch 1 | yes |
| D_la | logit-adjusted (LA) loss | RunPod (A40) | yes |
| E_bam_on_D_la | D + balanced neutral-vs-rest head (BAM, as in TG-MILNet) | Kaggle batch 2 | yes |
| F_ord_on_D_la | D + ordinal head (neg < neu < pos) | RunPod | yes |
| F_ord_on_B_strat (F*) | B + ordinal head | lab KC | yes |
| G_deskew_on_D_la | D + every frame sheared so the body axis is vertical | Kaggle batch 2 | yes |
| L_flip_on_D_la | D + horizontal flip augmentation | Kaggle batch 1 | yes |

Common protocol for all of them:
- Train 476/59/59 children (neg/neu/pos), val 120/15/15 (split from the official train set), official test 749.
- 16k iterations, lr drop at 10k, one seed (seed 0).
- The checkpoint, threshold (tau) and aux weight are chosen on val only. Test is never used for tuning.
- Sen = negative recall and Spe = non-negative called non-negative, as in ScoliGaitX and TG-MILNet. Balanced accuracy (bal) = mean of the 3 class recalls, which is the main metric because the test set is 73% scoliosis.

C_up and C_sqrt (upsampling variants) were planned but not run.

## 2. Suggested PPT flow (about 12 slides)

| # | slide | key message | figure / numbers |
|---|---|---|---|
| 1 | Title | | |
| 2 | Problem | Scoliosis screening from gait silhouettes; Scoliosis1K is 1:1:8-ish imbalanced; mild cases matter clinically | class counts |
| 3 | Reproduction | ScoNet-MT reproduces: 82.6 / 98.5 / 77.8 vs paper 82.0 / 99.0 / 76.5 | table |
| 4 | Diagnosis: neutral is the bottleneck | neutral recall 40.5; 97% of misses go to negative; AUC neu-vs-neg 0.95; 43% of training batches have no neutral | tau table (sec. 3.2) |
| 5 | Competitors sit on a threshold curve | ScoliGaitX (89.05/89.22/92.48) lies on ScoNet-MT's tau curve (tau 0.3: 89.3/91.7/90.1); TG-MILNet is above it | curve plot |
| 6 | exp_2 design | one protocol, val split, 8 variants, thresholds chosen on val | table in sec. 1 |
| 7 | Finding 1: threshold explains most gains | A goes 76.0 -> 87.3 bal with a val threshold; D/E/F advantage disappears | `fig1_threshold.png` |
| 8 | Finding 2: stratified sampler is the real gain | B 89.7 bal, +2.3 [+0.2, +4.5] vs A; others n.s. | `fig5_bootstrap.png` |
| 9 | Finding 3: neutral recall | B 86.5 vs A 81.0 vs TG-MILNet 70.6 (paper) | `fig2_neutral.png` |
| 10 | Finding 4: lean shortcut | mirror AUC 97 -> 27-36 for every non-flip model; Grad-CAM mostly off-body and follows lean | `fig3_mirror.png`, Grad-CAM figures |
| 11 | Finding 5: robustness trade-off | deskew: mirror AUC 59, -2 bal; flip: mirror AUC 94, -6.6 bal | `fig3_mirror.png` |
| 12 | Limitations + conclusion | 1 seed, small val, unstable checkpoints | `fig4_curves.png` |

## 3. Findings with the numbers to quote

### 3.1 Reproduction (checkpoint 1)
- ScoNet-MT, official pkl, 16k: acc 82.6 / Sen 98.5 / Spe 77.8 (paper 82.0 / 99.0 / 76.5). Per class: neg 98.5 / pos 87.6 / neu 40.5.
- Retraining on our own raw-segmented negatives gives 80.2 at 16k: no segmentation shortcut, the official pkl matches the paper.
- Two runs with different data disagree on 5-7% of identical children: single-run differences of 4-5 points are noise. No competitor paper reports seeds.

### 3.2 Diagnosis (checkpoint 2)
- The sampler picks children uniformly: a batch averages 6.4 neg / 0.8 neu / 0.8 pos, and 43% of batches contain no neutral.
- Neutral recall is 90.5 on train vs 40.5 on test: the model memorises the 74 training neutrals.
- The model ranks neutrals well (AUC neu-vs-neg 0.95-0.96), and AUC rises with training while neutral recall falls. Conclusion: the border is set by the class prior, not missing features.
- Post-hoc logit adjustment on the 16k model (tuned on test, for illustration only):

| tau | acc | Sen | Spe | neutral recall |
|---|---|---|---|---|
| 0 | 82.6 | 98.5 | 77.8 | 40.5 |
| 0.3 | 89.3 | 91.7 | 90.1 | 65.1 |
| 0.5 | 92.0 | 85.8 | 96.5 | 82.5 |

- ScoliGaitX lies on this curve, so its gain looks like a different operating point. TG-MILNet (89.9 / 99.5 / 86.4) is above it; in mini experiments its gain came from the BAM head.
- Mini-model experiments (notebooks 01-07) already hinted that LA loss, BAM and class triplet are equal once the threshold is fair, and that flip hurts. exp_2 tests this at full scale.

### 3.3 Finding 1: one val-chosen threshold closes most of the gap (`fig1_threshold.png`)

| variant | bal, default threshold | bal, val threshold | neutral recall (val threshold) |
|---|---|---|---|
| A ScoNet-MT | 76.0 | 87.3 | 81.0 |
| B stratified | 79.9 | **89.7** | **86.5** |
| D LA loss | 85.8 | 85.9 | 74.6 |
| E D + BAM | 88.4 | 87.6* | 84.9* |
| F D + ordinal | 88.8 | 89.2 | 85.7 |
| F* B + ordinal | 77.9 | 87.1 | 77.8 |
| G D + deskew | 74.3 | 83.7 | 74.6 |
| L D + flip | 83.3 | 80.7 | 73.8 |

\* E uses val-fuse (main head + BAM head, weight chosen on val).

- At the default threshold, D/E/F look 10-13 points better than A. With a val threshold, A catches up to within about 2 points.
- AUC scoliosis-vs-negative is 97.0-97.3 for A, B, D, E, F, F*: the models separate the classes equally well; the losses and heads mainly move the border.
- Message: imbalance methods must be compared at a calibrated operating point. Otherwise a border shift is reported as a better model.

### 3.4 Finding 2: the stratified sampler is the only training change that survives (`fig5_bootstrap.png`)

Paired bootstrap of balanced accuracy vs A_base (val threshold, 95% CI):

| variant | gain vs A |
|---|---|
| B stratified | **+2.3 [+0.2, +4.5]** (significant) |
| F D + ordinal | +1.9 [-0.0, +4.0] |
| E D + BAM (val-fuse) | +0.3 [-1.7, +2.2] |
| F* B + ordinal | -0.2 [-2.2, +1.8] |
| D LA loss | -1.4 [-3.7, +0.7] |
| G D + deskew | -3.6 [-6.2, -0.9] |
| L D + flip | -6.6 [-9.6, -3.5] |

- B vs D (the phase-2 base): +3.8 [+1.6, +6.2].
- Ordinal does not stack on B (F* -2.5 vs B): both act on the same thing, the neutral border.
- B with val threshold: acc 91.7 / Sen 86.8 / Spe 96.1, neutral recall 86.5. TG-MILNet (paper, full train set, 1 run): 89.9 / 99.5 / 86.4, neutral 70.6. Different train size (594 vs 744 children), so present it as a reference, not a win.

### 3.5 Finding 3: all models read the body lean (`fig3_mirror.png` + Grad-CAM)

Mirror test = flip every test video left/right; deskew test = remove the global lean.

| variant | AUC sco-vs-neg normal | mirrored | deskewed |
|---|---|---|---|
| A ScoNet-MT | 97.2 | 26.9 | 89.3 |
| B stratified | 97.2 | 30.6 | 91.1 |
| D LA loss | 97.0 | 26.8 | 90.8 |
| E D + BAM | 97.2 | 29.8 | 90.6 |
| F D + ordinal | 97.3 | 29.9 | 90.9 |
| F* B + ordinal | 97.3 | 36.5 | 91.4 |
| G D + deskew | 95.8 | **58.7** | 95.8 |
| L D + flip | 93.9 | **93.6** | 87.2 |

- An AUC far below 50 means the ranking is inverted: the models learned "leaning to one side = scoliosis".
- Deskewed input keeps AUC at about 90, so there is real signal beyond the lean. But that signal is still side-specific: deskew + mirror on ScoNet-MT gives 52.5.
- The lean differs by class in the raw silhouettes (upper-torso asymmetry 2.6 / 4.0 / 4.7 for neg / neu / pos) and alone gives AUC 81. It also varies between recording blocks of negatives (1.7-3.7), and class is fixed by subject-id block, so it may be partly a recording/session effect.
- Grad-CAM on ScoNet-MT 16k: only 28-40% of the scoliosis evidence lies on the body (which covers 35% of the image); legs and head get more than their share, torso and hips about half. Within negatives, the CAM side follows the lean (rho 0.63). The deletion test fails (CAM beats random in only 22-42% of children), so the heatmaps are not faithful explanations.
- Figures: `class_average_cam_sco_ScoNet-MT_16k.png`, `class_average_cam_true_ScoNet-MT_16k.png`, `examples_ScoNet-MT_16k.png`.

### 3.6 Finding 4: robustness costs accuracy
- G deskew: mirror AUC 27 -> 59 and mirror AUC neu-vs-neg 65 -> 85, at -2.1 bal vs D (n.s.) / -3.6 vs A (significant).
- L flip: fully mirror-invariant (93.6) but -6.6 bal vs A.
- Caveat for the write-up: deskew also removes any real trunk lean caused by scoliosis. Read the G result as "this much of the decision came from the global lean", not as "this much was a shortcut".

### 3.7 Training dynamics (`fig4_curves.png`)
- Every variant becomes good only after the lr drop at 10k. Before that, neutral recall jumps between 0 and 96.
- Even after 10k, neighbouring checkpoints differ a lot (A: 82 -> 70 -> 87 bal at 12k/14k/16k). Checkpoint choice is a source of variance that papers do not report.

## 4. Write-up structure

1. **Introduction**: screening motivation; mild cases matter; claims in the literature are reported at one uncalibrated operating point and one run. Contributions:
   1. A calibrated, val-based re-evaluation of imbalance fixes on Scoliosis1K.
   2. The finding that only class-stratified sampling gives a significant gain over a calibrated ScoNet-MT.
   3. Evidence that all models rely on the global body lean (mirror/deskew probes, Grad-CAM with faithfulness checks).
   4. Deskew/flip as robustness baselines with their accuracy cost.
2. **Related work**: ScoNet / ScoNet-MT, Gait-MIL, TG-MILNet, ScoliGaitX; class-imbalance methods (logit adjustment, balanced sampling, ordinal heads); shortcut learning and CAM faithfulness.
3. **Setup**: data and split (sec. 1), metric convention, val protocol, variants.
4. **Results**: sec. 3.3 and 3.4 (main table = val-threshold table + bootstrap).
5. **Analysis**: neutral errors (3.2), lean shortcut (3.5), robustness (3.6), training dynamics (3.7).
6. **Limitations**:
   - One seed. Earlier mini experiments showed a seed std of 2-4 points, so only gaps with a CI excluding 0 are claimed.
   - The val split has only 15 neutral children: one child moves val neutral recall by 6.7 points.
   - Reduced train set (594 vs 744 children), so absolute numbers are not comparable with papers.
   - Class is tied to subject-id blocks; a recording effect cannot be separated from a real posture difference with this data.
   - Deskew also removes a real scoliotic lean.
7. **Conclusion**: report calibrated operating points and AUC; use stratified sampling; evaluate with mirror/deskew probes.

## 5. What not to claim
- Do not say B beats TG-MILNet: different train size, one seed.
- Do not say LA/BAM/ordinal "do not work": they work at the default threshold. The claim is that their gain is mostly a border shift.
- Do not present Grad-CAM heatmaps as the model's explanation: the deletion test fails.
- Do not call the lean a pure dataset artefact: deskew input still gives AUC about 90, and the lean is partly real posture.
