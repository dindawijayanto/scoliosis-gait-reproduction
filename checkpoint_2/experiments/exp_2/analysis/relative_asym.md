# Relative trunk asymmetry from pose keypoints (camera-roll-free)

Written by `code/relative_asym.py`. Signed angles in degrees, median over usable front/back frames. "global" features tilt with the camera, "relative" ones are differences of two body lines and do not. Children: 1493; test 749; negatives used for the block check: 800 (all splits).

## 1. Single features: test AUC and the recording-block effect among healthy children

AUC > 50: higher value = more scoliotic; < 50: lower value = more scoliotic (the side matters, it is signed). eta^2 = % of the variance among all negative children explained by their 100-id block (session).

| feature | type | mean neg / neu / pos | AUC sco-neg | AUC neu-neg | block eta^2 (negatives) | p |
|---|---|---|---|---|---|---|
| shoulder_angle | global | +1.37 / +1.95 / +2.47 | 82.5 | 70.9 | 24.0 | 1e-42 |
| hip_angle | global | +1.13 / +1.64 / +1.99 | 85.3 | 75.9 | 24.1 | 7e-43 |
| trunk_lean | global | +0.98 / +1.51 / +1.87 | 87.0 | 75.5 | 22.5 | 2e-39 |
| shoulder_minus_hip | relative | +0.30 / +0.33 / +0.48 | 56.5 | 50.9 | 7.3 | 4e-10 |
| trunk_minus_hip | relative | -0.07 / -0.07 / -0.07 | 51.6 | 51.4 | 4.6 | 9e-06 |

## 2. Mean per 100-id block of the negatives (healthy children), against the class means

| group | n | shoulder_angle | hip_angle | trunk_lean | shoulder_minus_hip | trunk_minus_hip |
|---|---|---|---|---|---|---|
| negatives 400-499 | 7 | +0.77 | +0.79 | +0.58 | +0.03 | -0.16 |
| negatives 500-599 | 100 | +0.86 | +0.92 | +0.79 | +0.04 | -0.02 |
| negatives 600-699 | 100 | +0.88 | +0.86 | +0.77 | +0.10 | +0.01 |
| negatives 700-799 | 100 | +1.57 | +1.27 | +1.22 | +0.35 | +0.03 |
| negatives 800-899 | 100 | +1.20 | +0.92 | +0.81 | +0.32 | -0.02 |
| negatives 900-999 | 100 | +1.10 | +0.84 | +0.53 | +0.29 | -0.21 |
| negatives 1000-1099 | 100 | +1.90 | +1.54 | +1.43 | +0.45 | -0.06 |
| negatives 1100-1199 | 100 | +1.71 | +1.38 | +1.13 | +0.35 | -0.22 |
| negatives 1200-1299 | 93 | +1.83 | +1.39 | +1.19 | +0.53 | -0.07 |
| all negative | 800 | +1.37 | +1.13 | +0.98 | +0.30 | -0.07 |
| all neutral | 200 | +1.95 | +1.64 | +1.51 | +0.33 | -0.07 |
| all positive | 493 | +2.47 | +1.99 | +1.87 | +0.48 | -0.07 |

## 3. Logistic regression on the train children (3 classes), score = log p(neu or pos) - log p(neg)

Rolled = every test child gets its own random camera roll in [-2, 2] degrees (a different camera angle per session), mean AUC of 20 draws. Mirrored = left-right flipped test video. Block eta^2 = session effect in the scores of all negative children (train children included, so it is an in-sample view of the session effect).

| features | AUC sco-neg | AUC neu-neg | AUC sco-neg, random camera roll per child (+-2 deg) | AUC sco-neg, mirrored | block eta^2 of negatives |
|---|---|---|---|---|---|
| global (shoulder, hip, trunk lean) | 89.0 | 77.5 | 68.1 | 11.4 | 28.8 (p 1e-53) |
| relative (shoulder-hip, trunk-hip) | 56.4 | 51.2 | 56.4 | 43.5 | 6.2 (p 3e-08) |
| global + relative | 88.8 | 77.0 | 68.0 | 11.4 | 27.8 (p 3e-51) |
| DRF PAV (absolute, reference) | 85.9 | 77.3 | not computed | = normal (absolute values) | 17.2 (p 2e-28) |
