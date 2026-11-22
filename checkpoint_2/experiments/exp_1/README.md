# exp_1: validation-based calibration (never run)

The question: how much of ScoNet-MT's neutral failure can be fixed by choosing the checkpoint and the decision threshold on held-out data, with no test leakage?

**Status:** never trained. Its run (author config on the reduced train split) became `A_base` in `../exp_2`, which also uses the same val split. This folder is kept for the val-split script and the RunPod recipe.

## Files

| file | where it runs | what it does |
|---|---|---|
| `code/make_val_split.py` | local (already run) | Takes 120 negative / 15 neutral / 15 positive children out of the official train split, keeping the 1:1:8 ratio, with seed 0. Writes `partitions/`. |
| `partitions/Scoliosis1K_118_trainval.json` | pod | TRAIN = 594 reduced train children, TEST = 150 val children. |
| `partitions/Scoliosis1K_118_test.json` | pod | TRAIN = the same 594, TEST = the official 749 test children. |
| `code/setup_runpod_val.sh` | pod | Clones OpenGait, adds an `OPENGAIT_SEED` override, and writes the author config for seeds 0, 1 and 2. Only the partition and save_name change. |
| `code/export_logits.py` | pod | Writes `val_logits/<run>/{val,test}/iter_*.npz` for every checkpoint. |
| `code/calibrate.py` | local | Picks the checkpoint and tau on val, then reports on test (`analysis/calibration_summary.json`). `--smoke` reruns the old test-tuned numbers as a code check. |

`../exp_2/partitions/` holds the same two split files.

## Steps (if it were run)

1. Upload to `/workspace`: `code/setup_runpod_val.sh`, `code/export_logits.py`, both files from `partitions/`, and `Scoliosis1K-sil-pkl.zip`.
2. Run `bash /workspace/setup_runpod_val.sh`, then train the 3 seeds using the commands it prints.
3. Run `python /workspace/export_logits.py` from `/workspace/OpenGait`.
4. Download `/workspace/val_logits` to `runs/val_logits/` here, then run `python code/calibrate.py` locally.

## Caveats

- **Smaller train set:** the model trains on 59 neutral children, not 74. Compare `fixed16k` against the val rules within these runs; do not compare against the old full-train run.
- **Small val set:** 15 val neutrals means each one moves val neutral recall by 6.7 points, so tau is noisy. Check how much the chosen tau varies across seeds.
- **Default rule:** balanced accuracy, which does not assume a class prior. `val-tau@Sen97` adds a constraint that keeps negative recall near the papers' 99.
