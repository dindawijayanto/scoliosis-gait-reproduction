# PAV from pose keypoints + late fusion with B_strat_s0 @ 16000

Written by `code/pav_fusion.py`. PAV as in DRF (MICCAI 2025): 8 symmetric COCO pairs x (vertical, midline, angle) differences, hip centre at the origin, height 1, IQR outlier removal, mean over frames. "+ std" adds the spread over frames. Absolute values: mirror-invariant by construction.

## 1. Single PAV features on test (AUC, scoliosis vs negative / neutral vs negative)

| feature | AUC sco-neg | AUC neu-neg |
|---|---|---|
| ankle_mid | 76.5 | 72.6 |
| knee_mid | 73.6 | 62.5 |
| shoulder_mid | 72.1 | 62.8 |
| shoulder_ang | 71.2 | 65.5 |
| shoulder_vert | 70.9 | 63.8 |
| hip_vert | 70.1 | 64.9 |
| hip_ang | 69.4 | 64.9 |
| ankle_ang | 68.6 | 70.9 |

## 2. PAV alone and fused with B_strat_s0 (test; tau, lam chosen on val)

| model | tau | lam | acc | Sen | Spe | rec_neu | rec_pos | bal | AUC_sn | AUC_nn | mirror AUC_sn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PAV mean (DRF) alone | - | - | 60.9 | 66.7 | 86.1 | 44.4 | 63.0 | 58.0 | 86.6 | 80.3 | (same) |
| B_strat_s0 val-tau | 0.6 | 0 | 91.7 | 86.8 | 96.1 | 86.5 | 95.7 | 89.7 | 97.2 | 94.2 | 30.6 |
| B_strat_s0 + PAV mean (DRF) | 0.35 | 0.15 | 90.5 | 92.6 | 92.8 | 75.4 | 94.0 | 87.4 | 98.1 | 95.3 | 60.8 |
| PAV mean + std over time alone | - | - | 64.2 | 74.5 | 83.7 | 44.4 | 65.2 | 61.4 | 89.1 | 80.8 | (same) |
| B_strat_s0 + PAV mean + std over time | 0.3 | 0.05 | 89.7 | 93.1 | 90.6 | 74.6 | 92.6 | 86.8 | 97.7 | 94.9 | 44.9 |

## 3. Paired bootstrap of balanced accuracy vs B_strat_s0 val-tau (2000 resamples of test children)

- B_strat_s0 + PAV mean (DRF): -2.3 [-4.4, -0.2]
- B_strat_s0 + PAV mean + std over time: -2.9 [-5.4, -0.6]

## 4. Recording-block check (scoliosis score of a PAV-only logistic, trained on the train children)

Mean score per block of negative subject ids, against the class means on test. If the negative blocks spread as much as the classes, PAV also carries a recording effect.

| group | n | mean score |
|---|---|---|
| negatives, ids 400-499 | 7 | -1.69 |
| negatives, ids 500-599 | 100 | -1.76 |
| negatives, ids 600-699 | 100 | -1.84 |
| negatives, ids 700-799 | 100 | -1.01 |
| negatives, ids 800-899 | 100 | -1.43 |
| negatives, ids 900-999 | 100 | -1.56 |
| negatives, ids 1000-1099 | 100 | -0.00 |
| negatives, ids 1100-1199 | 100 | -0.49 |
| negatives, ids 1200-1299 | 93 | -0.35 |
| test negative | 204 | -1.06 |
| test neutral | 126 | +0.54 |
| test positive | 419 | +1.60 |
