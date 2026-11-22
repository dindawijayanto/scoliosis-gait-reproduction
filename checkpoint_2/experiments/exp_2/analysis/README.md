# exp_2 / analysis

| file | what | written by |
|---|---|---|
| `runs.md` | one section per training run: what changed, where it ran, fixed vs val-calibrated test, probes, learning curve, reading | by hand from the tables below |
| `STORY.md` | the results story for the PPT and write-up: findings in order, numbers to quote, what not to claim | by hand |
| `summary.md`, `summary.json` | test tables under the fixed / val-tau / val-fuse rules, bootstrap vs `D_la`, AUC on normal / mirrored / deskewed test, 5-NN, learning curves | `code/calibrate.py --ref D_la` |
| `curves.json` | val and test balanced accuracy, neutral recall and AUC per checkpoint | `code/calibrate.py` |
| `bootstrap.md` | paired bootstrap of balanced accuracy vs `A_base` | `code/make_figs.py` |
| `figs/` | `fig1_threshold` (fixed vs val tau), `fig2_neutral`, `fig3_mirror` (lean probes), `fig4_curves`, `fig5_bootstrap` | `code/make_figs.py` |
| `pav_fusion_<run>.md` | PAV (DRF-style postural asymmetry from the pose keypoints): single-feature AUCs, PAV alone, late fusion with the run (tau and weight chosen on val), bootstrap, recording-block check | `code/pav_fusion.py --run <run> --iter <it>` |
| `relative_asym.md` | signed global tilts (shoulder, hip, trunk lean) vs camera-roll-free relative tilts (shoulder - hip, trunk - hip) from the pose keypoints: test AUC, recording-block effect among healthy children, random-roll and mirror checks | `code/relative_asym.py` |
| `gradcam/gradcam.ipynb` | Grad-CAM on ScoNet-MT 16k and the exp_2 runs: in-body share, band shares, CAM side vs lean, deletion and sanity checks; figures in `gradcam/output/figures/` | notebook |

Rerun `calibrate.py` and `make_figs.py` after adding runs to `../runs/logits/`.
