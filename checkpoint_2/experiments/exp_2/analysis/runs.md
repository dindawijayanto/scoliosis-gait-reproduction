# exp_2: analysis per training run

One section per trained run (seed 0, 16k iterations). Numbers come from `summary.md` (written by `code/calibrate.py --ref D_la`) and `bootstrap.md` (written by `code/make_figs.py`, reference `A_base`).

How to read each section:
- **fixed** = tau 0 at the last checkpoint, the way papers report.
- **val** = checkpoint and tau (and aux weight for runs with a head) chosen on the val split by balanced accuracy, then test reported once.
- **bal** = mean of the three class recalls (main metric). Sen = negative recall, Spe = scoliosis not called negative (paper convention).
- **AUC_sn** = scoliosis vs negative, **AUC_nn** = neutral vs negative, on the normal / mirrored / deskewed test set. Mirror AUC far below 50 means the ranking flips when the video is mirrored, so the model reads the left/right lean.
- **curve** = test bal / neutral recall per checkpoint, with tau chosen on val at that checkpoint.

Overview (val rule, val-fuse for E):

| run | bal | neutral recall | vs A_base [95% CI] | AUC_sn | mirror AUC_sn |
|---|---|---|---|---|---|
| A_base | 87.3 | 81.0 | - | 97.2 | 26.9 |
| B_strat | **89.7** | **86.5** | **+2.3 [+0.2, +4.5]** | 97.2 | 30.6 |
| D_la | 85.9 | 74.6 | -1.4 [-3.7, +0.7] | 97.0 | 26.8 |
| E_bam_on_D_la | 87.6 | 84.9 | +0.3 [-1.7, +2.2] | 97.1 | 29.8 |
| F_ord_on_D_la | 89.2 | 85.7 | +1.9 [-0.0, +4.0] | 97.3 | 29.9 |
| F_ord_on_B_strat | 87.1 | 77.8 | -0.2 [-2.2, +1.8] | 97.3 | 36.5 |
| G_deskew_on_D_la | 83.7 | 74.6 | -3.6 [-6.2, -0.9] | 95.8 | **58.7** |
| L_flip_on_D_la | 80.7 | 73.8 | -6.6 [-9.6, -3.5] | 93.9 | **93.6** |

---

## A_base_s0: ScoNet-MT, the reference

- **Change:** none. Natural TripletSampler (8 random children of 594, so 43.8% of batches have no neutral), CE loss.
- **Trained on:** lab RTX 3060, resumed from 12k (`restore_hint: 12000` in its config).
- **fixed:** acc 80.8 / Sen 97.1 / Spe 76.3, neutral 48.4, bal 76.0.
- **val (16k, tau 0.4):** acc 89.7 / Sen 87.7 / Spe 93.9, neutral 81.0, bal 87.3.
- **AUC_sn** 97.2 / mirror 26.9 / deskew 89.3. **AUC_nn** 94.3 / mirror 62.1 / deskew 87.6.
- **curve:** 8k 56/85, 10k 59/0, 12k 82/81, 14k 70/27, 16k 87/81.

**Reading.** One val-chosen tau lifts bal by 11.3 points without retraining. Same diagnosis as checkpoint 2: the ranking is good (AUC 97), the border is set by the 80/10/10 training prior. The curve is unstable even after the lr drop (82 -> 70 -> 87), so the checkpoint choice is a source of variance. Mirror AUC 26.9 means the ranking is inverted on mirrored input.

## B_strat_s0: stratified sampler

- **Change:** `ClassAwareSampler(mode='stratified')`: 2 children per class per batch plus 2 random extra slots, so every batch has all three classes. CE (LA would be the same here, because the sampler prior is uniform).
- **Trained on:** Kaggle batch 1.
- **fixed:** acc 84.6 / Sen 96.6 / Spe 81.1, neutral 55.6, bal 79.9.
- **val (16k, tau 0.6):** acc 91.7 / Sen 86.8 / Spe 96.1, neutral 86.5, bal 89.7.
- **AUC_sn** 97.2 / mirror 30.6 / deskew 91.1. **AUC_nn** 94.2 / mirror 67.8 / deskew 89.3. 5-NN embedding bal 53.9.
- **curve:** 8k 61/19, 10k 81/83, 12k 82/75, 14k 88/89, 16k 90/87.
- **Bootstrap:** +2.3 [+0.2, +4.5] vs A, +3.8 [+1.6, +6.2] vs D. The only training change whose CI excludes 0.

**Reading.** The best run and the only significant gain. AUC is the same as A, so it is not a better ranking of two classes; the gain comes from the three-way decision after calibration and from a stabler end of training (the only run above 80 bal at every checkpoint from 10k on: 81, 82, 88, 90). It still needs tau 0.6 although its batches are balanced. A plausible reason is that the 59 training neutrals are memorised, so neutral scores on new children are lower than in training.

## D_la_s0: logit-adjusted loss

- **Change:** `LogitAdjustedCE` with tau 1: the loss sees `16*logits + log(prior)`, so the prior does not have to be stored in the logits.
- **Trained on:** RunPod A40.
- **fixed:** acc 89.7 / Sen 81.9 / Spe 96.9, neutral 78.6, bal 85.8.
- **val (14k, tau -0.25):** acc 89.7 / Sen 87.7 / Spe 93.9, neutral 74.6, bal 85.9.
- **AUC_sn** 97.0 / mirror 26.8 / deskew 90.8. **AUC_nn** 94.5 / mirror 65.3 / deskew 88.9. 5-NN 50.4.
- **curve:** 8k 52/71, 10k 58/0, 12k 62/96, 14k 86/75, 16k 86/76.

**Reading.** At tau 0 it looks 10 points better than A, but after val calibration it is 1.4 behind (n.s.). Its val tau is near 0, as the theory says: LA does at training time what tau does after training. So LA's gain in the fixed table is a border shift, not a better model. It was used as the phase-2 base before B's result was known.

## E_bam_on_D_la_s0: BAM neutral head

- **Change:** D plus a second `SeparateBNNecks` with 2 classes (neutral vs rest), trained with LA on its own binary prior. At test its score can be fused into the neutral logit with weight lam chosen on val.
- **Trained on:** Kaggle batch 2.
- **fixed:** acc 90.4 / Sen 88.2 / Spe 94.1, neutral 83.3, bal 88.4.
- **val-tau (14k, tau 0.1):** bal 86.4, neutral 86.5. **val-fuse (16k, tau 0.05, lam 0.25):** acc 89.5 / Sen 84.8 / Spe 95.2, neutral 84.9, bal 87.6.
- **AUC_sn** 97.1 / mirror 29.8 / deskew 90.6. **AUC_nn** 95.3 (best) / mirror 70.5 / deskew 90.4. 5-NN 51.3.
- **curve:** 8k 49/65, 10k 71/71, 12k 69/79, 14k 86/87, 16k 88/85.

**Reading.** The best fixed-threshold run, which is the TG-MILNet story: BAM helps a lot when nobody calibrates. After calibration it is +0.3 vs A (n.s.). It has the highest AUC_nn, so the head does sharpen neutral vs negative a little, but not enough to change balanced accuracy.

## F_ord_on_D_la_s0: ordinal head

- **Change:** D plus two binary heads, `ord1` = y >= neutral and `ord2` = y >= positive (negative < neutral < positive). `ord1` learns from all 118 scoliosis children.
- **Trained on:** RunPod A40.
- **fixed:** acc 90.4 / Sen 84.3 / Spe 96.3, neutral 88.1, bal 88.8.
- **val-tau (16k, tau -0.05):** acc 90.9 / Sen 88.2 / Spe 95.2, neutral 85.7, bal 89.2. **val-fuse (14k, lam 0.5):** bal 87.5.
- **AUC_sn** 97.3 / mirror 29.9 / deskew 90.9. **AUC_nn** 94.2 / mirror 65.1 / deskew 89.2. 5-NN 52.7.
- **curve:** 8k 60/0, 10k 57/0, 12k 81/78, 14k 82/87, 16k 89/86.
- **Bootstrap:** +1.9 [-0.0, +4.0] vs A (val-tau), borderline.

**Reading.** The second-best run. With an oracle test tau its ceiling is 89.2 vs 86.2 for D (RunPod analysis), so the ordinal head does more than move the border on top of LA. Val picked fusion with lam 0.5, which was worse on test (87.5 vs 89.2): with 15 val neutrals the aux weight choice is noisy.

## F_ord_on_B_strat_s0: ordinal head on the stratified sampler

- **Change:** B plus the ordinal head (CE, because the prior is uniform).
- **Trained on:** lab RTX 3060. The config was not saved and was rebuilt in `configs/`.
- **fixed:** acc 80.9 / Sen 98.0 / Spe 75.6, neutral 55.6, bal 77.9.
- **val (16k, tau 0.55):** acc 89.5 / Sen 91.7 / Spe 91.6, neutral 77.8, bal 87.1.
- **AUC_sn** 97.3 / mirror 36.5 / deskew 91.4. **AUC_nn** 94.1 / mirror 73.2 / deskew 89.2.
- **curve:** 8k 70/63, 10k 70/24, 12k 88/80, 14k 88/87, 16k 87/78.
- **Bootstrap:** -2.5 [-4.6, -0.5] vs B.

**Reading.** The ordinal head does not stack on B. AUC is unchanged, so ordinal and stratified sampling act on the same thing, the neutral border; once B has fixed it, the head only moves it again. Its mirror AUC (36.5) is the least inverted of the non-robust runs, but the lean dependence is still clearly there.

## G_deskew_on_D_la_s0: global lean removed

- **Change:** D with `DeskewSilCuttingTransform` in training and test: every frame is sheared so its best-fit body axis is vertical.
- **Trained on:** Kaggle batch 2.
- **fixed:** acc 80.9 / Sen 44.6 / Spe 99.8, neutral 79.4, bal 74.3.
- **val (14k, tau -0.45):** acc 85.7 / Sen 89.2 / Spe 87.5, neutral 74.6, bal 83.7.
- **AUC_sn** 95.8 / mirror **58.7** / deskew 95.8. **AUC_nn** 93.6 / mirror **85.2** / deskew 93.6. 5-NN 48.1.
- **curve:** 8k 45/69, 10k 39/0, 12k 81/89, 14k 84/75, 16k 79/75.
- **Bootstrap:** -3.6 [-6.2, -0.9] vs A, -2.1 [-4.8, +0.6] vs D.

**Reading.** The key robustness result. Every non-deskew model has mirror AUC 27-37; this one has 58.7, so most of the dependence on the global lean is gone, at a cost of about 2 bal vs its base D. 58.7 is still far below its normal 95.8, so some side-specific signal remains beyond the global lean. At tau 0 it calls more than half of the negatives scoliosis (Sen 44.6): without the lean, the LA-trained border sits in a very different place, which val calibration fixes. Deskew also removes any real scoliotic trunk lean, so the drop is "the part of the decision that came from the global lean", not "the shortcut".

## L_flip_on_D_la_s0: horizontal flip in training

- **Change:** D with `train_flip: true`: whole training sequences mirrored with probability 0.5.
- **Trained on:** Kaggle batch 1.
- **fixed:** acc 85.3 / Sen 75.0 / Spe 94.7, neutral 84.1, bal 83.3.
- **val (14k, tau -0.25):** acc 81.7 / Sen 86.8 / Spe 85.1, neutral 73.8, bal 80.7.
- **AUC_sn** 93.9 / mirror **93.6** / deskew 87.2. **AUC_nn** 93.0 / mirror 93.1 / deskew 90.4. 5-NN 45.4.
- **curve:** 8k 53/33, 10k 48/0, 12k 62/67, 14k 81/74, 16k 84/81.
- **Bootstrap:** -6.6 [-9.6, -3.5] vs A.

**Reading.** Fully mirror-invariant, as expected, and the largest accuracy loss, as notebook 08 predicted. The left/right cue carries a lot of accuracy on this test set. Val chose 14k although 16k is better on test, which costs part of the gap.

---

## Across runs

1. **Threshold first.** At tau 0, D/E/F look 10-13 points better than A. After val calibration A is within about 2 points of all of them. AUC_sn is 97.0-97.3 for A, B, D, E, F and F*.
2. **Only the stratified sampler survives calibration** (+2.3 vs A, CI excludes 0).
3. **Every non-robust model reads the lean** (mirror AUC 27-37). Deskew halves this (58.7) for -2 bal; flip removes it (93.6) for -6.6 bal.
4. **Training is jumpy.** Nothing is good before the lr drop at 10k (neutral recall jumps between 0 and 96), and neighbouring checkpoints can differ by 10+ points after it.
5. **Embeddings are not organised by class** (5-NN bal 45-54 everywhere): the triplet loss uses child ID, so the class information lives in the classifier, not in the embedding geometry.

---

## Add-on (2026-11-21): PAV from pose keypoints, late fusion (no training)

`code/pav_fusion.py`. PAV = DRF's postural asymmetry vector (MICCAI 2025): |left-right| height, midline and angle differences of 8 symmetric COCO keypoint pairs, mean (and std) over frames. Mirror-invariant by construction. A logistic regression on the train children gives log p(class), added to the run's logits with a weight chosen on val.

| | bal | neutral | AUC_sn | mirror AUC_sn | vs the run alone |
|---|---|---|---|---|---|
| PAV alone (mean + std) | 61.4 | 44.4 | 89.1 | 89.1 | - |
| B_strat_s0 | 89.7 | 86.5 | 97.2 | 30.6 | - |
| B_strat_s0 + PAV (lam 0.15) | 87.4 | 75.4 | 98.1 | 60.8 | -2.3 [-4.4, -0.2] |
| G_deskew_on_D_la_s0 @16k | 78.7 | 75.4 | 95.8 | 58.7 | - |
| G_deskew_on_D_la_s0 + PAV (lam 0.1-0.2) | 82.7-83.0 | 74 | 97.3-97.7 | 77-82 | +4.0 to +4.4 (CI > 0) |

**Reading.**
- On its own, PAV ranks much better than the silhouette posture features: AUC 87-89 vs 60-71.
- Fused, it raises AUC and mirror robustness for every model.
- It only raises balanced accuracy for the deskewed model, i.e. the model that has no lean shortcut left. For B_strat, the extra signal moves the neutral border and costs 2.3 points.
- PAV also varies between recording blocks of negative ids: blocks 1000-1299 score close to the neutral class mean (section 4 of `pav_fusion_*.md`). So even this side-free clinical prior may carry a recording effect, which DRF does not test.

---

## Phase 3 (2026-11-21): order-aware temporal models (RunPod 3x A40)

Full tables: `temporal.md` (written by `code/temporal_report.py`), PAV fusion: `pav_fusion_S_*.md`.

| model | what | seeds | bal (val tau) | vs reference |
|---|---|---|---|---|
| B_strat_s0 | reference: random frames, max | 1 | **89.7** | - |
| T_hppmax | control: fine-tune 4k on ordered clips, order-free max | 3 | 86.2 +- 1.8 | -2.7 vs B (seed 0) |
| T_tconv | + temporal conv | 3 | 84.8 +- 3.3 | -1.4 [-2.6, -0.1] vs control; better on 1/3 seeds |
| T_pose | + temporal conv + mirror-invariant posture stream | 3 | 77.1 +- 0.8 | -9.1 [-11.3, -7.0] vs control; AUC 97 -> 92 |
| S_tconv | from scratch 16k, ordered clips + temporal conv | 1 | 81.6 | -8.0 [-11.1, -5.0] vs B |
| S_pose | from scratch, + posture stream | 1 | 77.1 | -12.6 [-15.7, -9.2] vs B |
| S_tconv + pose PAV | late fusion, weight on val | 1 | 83.1 | +1.5 vs S_tconv |

**Reading.**
- An order-aware temporal model does not help on Scoliosis1K. This holds against the right control and over 3 seeds, and also when the temporal model is trained from scratch.
- The from-scratch model really uses the frame order: bal drops from 81.6 to 77.3 on shuffled frames. The fine-tuned one does not (85.1 vs 84.9). So order can be learned, but it carries no extra class signal here.
- The posture stream lowers the lean dependence (mirror AUC 51-60 vs 31-42) but costs ranking quality (AUC 92 vs 97). It is the same robustness-accuracy trade-off as deskew and flip.
- This agrees with DeepGaitV2: temporal modelling adds little on constrained indoor gait data. It also agrees with notebook 04, which tested order-free pooling.
- Raw accuracy at tau 0 looked good for S_tconv (87.1 vs 84.6 for B_strat). Calibrated balanced accuracy shows the opposite. This is again the accuracy-on-a-73%-scoliosis-test trap.

### Logit ensembles (2026-11-21, no training; tau on val)

| ensemble | members | bal | AUC_sn | vs B_strat_s0 |
|---|---|---|---|---|
| B_strat_s0 alone | 1 | **89.7** | 97.2 | - |
| B_strat checkpoints 12/14/16k | 3 | 88.2 | 96.8 | -1.5 [-2.7, -0.2] |
| T_hppmax + T_tconv, all seeds | 6 | 88.2 | 96.9 | -1.4 [-3.2, +0.4] |
| B + 6 fine-tunes + S_tconv | 8 | 88.4 | 97.1 | -1.3 [-3.1, +0.7] |
| exp_2 zoo A, B, D, E, F, F* | 6 | 87.5 | 97.4 | -2.2 [-4.2, -0.4] |

**Reading.**
- No ensemble beats the single B_strat_s0.
- AUC stays at 97 whatever is combined: the ranking is saturated on 64x44 silhouettes.
- The fine-tunes of B_strat_s0 average 86.2 (val tau) / 88.2 (oracle tau). So 89.7 is probably the upper end of B_strat's seed spread, and B_strat needs more seeds before 89.7 is quoted as its typical value.

---

## B_strat seeds 1 and 2 (2026-11-22, Kaggle T4 x2)

- **What changed:** nothing but the seed (`OPENGAIT_SEED` 1 and 2). The configs are `B_strat_s0` with only `save_name` changed. Trained from branch `exp2-kaggle` with `code/kaggle_Bseeds.ipynb`; val/test logits at 12k/14k/16k, mirror and deskew probes at 16k.
- **Rule:** every seed picks checkpoint and tau on val among 12k/14k/16k (seed 0 is restricted to the same pool, which does not change its pick).

| seed | iter | tau | bal | neg | neu | pos | oracle bal | AUC_sn | AUC_nn | mirror AUC | deskew AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16k | 0.60 | 89.7 | 86.8 | 86.5 | 95.7 | 89.7 | 97.2 | 94.2 | 30.6 | 91.1 |
| 1 | 16k | 0.60 | 89.2 | 86.3 | 84.9 | 96.4 | 89.2 | 97.3 | 94.5 | 30.8 | 92.0 |
| 2 | 14k | 0.30 | 86.9 | 92.6 | 77.0 | 91.2 | 88.0 | 97.1 | 94.3 | 34.8 | 91.6 |
| mean | | | **88.6 +- 1.2** | | 82.8 +- 4.2 | | 89.0 | 97.2 | | 32.1 | |

Paired bootstrap of balanced accuracy vs `A_base_s0` (val tau, 2000 resamples):

| comparison | gain | 95% CI |
|---|---|---|
| B seed 0 vs A | +2.3 | [+0.2, +4.5] |
| B seed 1 vs A | +1.9 | [-0.0, +3.8] |
| B seed 2 vs A | -0.4 | [-2.9, +2.1] |
| B, mean of 3 seeds vs A | **+1.3** | **[-0.5, +3.2]** (B ahead in 91.7% of the resamples) |

**Reading.**
- 89.7 was the top of B's seed spread. Seed 2 is genuinely weaker (about 86.9 at every checkpoint), not a bad tau pick.
- The "only significant gain" of Finding 2 does not survive over seeds: B is still the best variant on average, but +1.3 is within noise. A has one seed only, so the fair comparison needs A seeds too.
- Ranking (AUC 97.1-97.3) and the lean inversion (mirror AUC 31-35) hold in every seed.
- At a screening border (at least 95% of the sick children found on val), seeds 0 and 1 miss 21-22 of 545 sick children vs 33 for A, with about 13% of healthy children referred.

## Relative trunk asymmetry from the pose keypoints (2026-11-21, no training)

`code/relative_asym.py` -> `relative_asym.md`. Signed angles, median over front/back frames.

- **Global tilts carry the class signal:** shoulder line / hip line / trunk axis lean +1.0 to +1.4 deg (negative) vs +1.9 to +2.5 deg (positive); each alone gives AUC 82-87, a logistic model on the three 89.
- **Relative tilts carry almost none:** shoulder minus hip AUC 56.5; trunk minus hip is -0.07 deg in every class. These do not change under a camera roll (checked by rotating the keypoints).
- **Session effect:** the 100-id block explains 22-24% of the global-tilt variance among the 800 healthy children (relative: 5-7%). Healthy blocks range from +0.86 deg (ids 500-699) to +1.90 deg (ids 1000-1099), almost the negative-to-positive gap.
- **Fragile:** a random camera roll of +-2 deg per child drops the global logistic from AUC 89 to 68; mirrored video gives 11.

**Reading.** What separates the classes is a rigid whole-body tilt (camera or stance), not a differential trunk deformation. It explains why the temporal models gain nothing (the cue is static, visible in every frame) and why every architecture is hard on the same recording blocks. Caveat: keypoints are coarse, so a real shoulder/hip difference under 0.5 deg may be lost.
