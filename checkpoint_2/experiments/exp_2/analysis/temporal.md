# exp_2 phase 3: order-aware temporal models

Written by `code/temporal_report.py`. Test = official 749 children; checkpoint and tau chosen on val (120/15/15). T_* runs are 4k-iteration fine-tunes of B_strat_s0@16k on ordered 30-frame clips, S_* runs are trained from scratch like B_strat_s0 (16k) on ordered clips. One row per run.

## 1. Every run (val-tau rule)

| run | iter | tau | acc | Sen | Spe | rec_neu | rec_pos | bal | AUC_sn | AUC_nn |
|---|---|---|---|---|---|---|---|---|---|---|
| B_strat_s0 | 16k | 0.6 | 91.7 | 86.8 | 96.1 | 86.5 | 95.7 | 89.7 | 97.2 | 94.2 |
| S_pose_strat_s0 | 16k | -0.25 | 79.2 | 98.0 | 72.7 | 56.3 | 76.8 | 77.1 | 96.5 | 94.2 |
| S_tconv_strat_s0 | 16k | -0.2 | 84.6 | 93.6 | 83.9 | 65.1 | 86.2 | 81.6 | 95.9 | 93.1 |
| T_hppmax_on_B_strat_s0 | 4k | 0.35 | 88.4 | 86.8 | 92.7 | 83.3 | 90.7 | 86.9 | 97.0 | 94.4 |
| T_hppmax_on_B_strat_s1 | 4k | 0.05 | 87.2 | 94.1 | 86.8 | 67.5 | 89.7 | 83.8 | 97.0 | 94.4 |
| T_hppmax_on_B_strat_s2 | 3k | 0.45 | 89.9 | 85.3 | 94.7 | 84.9 | 93.6 | 87.9 | 97.0 | 94.3 |
| T_pose_on_B_strat_s0 | 3k | 0.35 | 79.7 | 84.8 | 83.7 | 60.3 | 83.1 | 76.1 | 91.9 | 88.3 |
| T_pose_on_B_strat_s1 | 3k | 0.7 | 81.8 | 74.5 | 90.6 | 69.8 | 89.0 | 77.8 | 91.8 | 87.5 |
| T_pose_on_B_strat_s2 | 3k | 0.45 | 82.4 | 88.2 | 84.0 | 57.1 | 87.1 | 77.5 | 92.7 | 89.6 |
| T_tconv_on_B_strat_s0 | 3k | -0 | 87.4 | 85.8 | 93.0 | 84.9 | 89.0 | 86.6 | 96.4 | 94.4 |
| T_tconv_on_B_strat_s1 | 3k | -0.1 | 87.0 | 88.2 | 93.4 | 88.9 | 85.9 | 87.7 | 96.7 | 94.3 |
| T_tconv_on_B_strat_s2 | 4k | -0.35 | 81.7 | 95.1 | 78.9 | 65.1 | 80.2 | 80.1 | 96.2 | 93.8 |

## 2. Probes at the last checkpoint (val tau kept)

shuffle = the test frames in a random order. An order-free model gives exactly the normal numbers; a drop means the model uses the order of the frames. mirror / deskew as in exp_2 (lean shortcut).

| run | bal normal | bal shuffled | AUC_sn normal | AUC_sn shuffled | AUC_sn mirror | AUC_sn deskew |
|---|---|---|---|---|---|---|
| B_strat_s0 | 89.7 | - | 97.2 | - | 30.6 | 91.1 |
| S_pose_strat_s0 | 77.1 | 73.9 | 96.5 | 95.2 | 51.0 | 91.0 |
| S_tconv_strat_s0 | 81.6 | 77.3 | 95.9 | 94.7 | 39.3 | 91.3 |
| T_hppmax_on_B_strat_s0 | 86.9 | 86.9 | 97.0 | 97.0 | 41.9 | 92.7 |
| T_hppmax_on_B_strat_s1 | 83.8 | 83.8 | 97.0 | 97.0 | 41.7 | 93.0 |
| T_hppmax_on_B_strat_s2 | 87.4 | 87.4 | 96.9 | 96.9 | 41.1 | 92.1 |
| T_pose_on_B_strat_s0 | 75.0 | 72.1 | 92.0 | 91.0 | 57.8 | 80.3 |
| T_pose_on_B_strat_s1 | 77.2 | 73.3 | 91.9 | 91.8 | 57.2 | 80.9 |
| T_pose_on_B_strat_s2 | 77.7 | 71.0 | 92.7 | 87.7 | 60.0 | 81.0 |
| T_tconv_on_B_strat_s0 | 84.9 | 85.1 | 96.4 | 96.3 | 36.6 | 91.5 |
| T_tconv_on_B_strat_s1 | 83.4 | 82.8 | 96.8 | 96.3 | 38.0 | 91.3 |
| T_tconv_on_B_strat_s2 | 80.1 | 83.2 | 96.2 | 96.2 | 34.8 | 90.8 |

## 3. Temporal vs control over seeds (fine-tunes of the same B_strat_s0; val-tau bal)

| model | seeds | bal per seed | mean +- std | minus control, per seed | seeds better | bootstrap vs control [95% CI] |
|---|---|---|---|---|---|---|
| T_hppmax | 0,1,2 | 86.9, 83.8, 87.9 | 86.2 +- 1.8 | - | - | - |
| T_tconv | 0,1,2 | 86.6, 87.7, 80.1 | 84.8 +- 3.3 | -0.4, +3.9, -7.8 | 1/3 | -1.4 [-2.6, -0.1] |
| T_pose | 0,1,2 | 76.1, 77.8, 77.5 | 77.1 +- 0.8 | -10.9, -6.0, -10.4 | 0/3 | -9.1 [-11.3, -7.0] |

## 4. vs B_strat_s0 (S_*: same sampler and schedule from scratch, but ordered clips + temporal conv; T_*_s0: fine-tunes of it)

- S_pose_strat_s0: -12.6 [-15.7, -9.2] balanced accuracy vs B_strat_s0
- S_tconv_strat_s0: -8.0 [-11.1, -5.0] balanced accuracy vs B_strat_s0
- T_hppmax_on_B_strat_s0: -2.7 [-4.7, -0.8] balanced accuracy vs B_strat_s0
- T_pose_on_B_strat_s0: -13.6 [-17.1, -10.1] balanced accuracy vs B_strat_s0
- T_tconv_on_B_strat_s0: -3.1 [-5.2, -0.9] balanced accuracy vs B_strat_s0

Reference: B_strat_s0 val-tau 89.7 bal; TG-MILNet (paper, full train set, 1 run) acc 89.9 / Sen 99.5 / Spe 86.4 / neutral 70.6.
