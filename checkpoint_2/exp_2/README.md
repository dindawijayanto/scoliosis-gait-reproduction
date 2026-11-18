# Checkpoint 2 / exp_2: full training on the lab GPU

Self-contained copy of OpenGait (commit `4d228e1`) with the exp_2 additions, like `reproduce/model_sconet/` in checkpoint 1.
Everything runs **from this folder** (`checkpoint_2/exp_2/`), which plays the role of the OpenGait repo root.

Added to OpenGait:
- `opengait/data/sampler.py` -> `ClassAwareSampler` (stratified / upsample)
- `opengait/data/transform.py` -> `DeskewSilCuttingTransform`
- `opengait/modeling/losses/logit_adjusted.py` -> `LogitAdjustedCE`
- `opengait/modeling/models/sconet_robust.py` -> `ScoNetRobust` (BAM / ordinal head)
- `opengait/main.py` -> `OPENGAIT_SEED` override

The experiment itself (variants, steps, caveats) is described in [`exp_2/README.md`](exp_2/README.md).

```bash
cd ~/<group>/scoliosis-gait-reproduction/checkpoint_2/exp_2
ZIP=~/<group>/Scoliosis1K-sil-pkl.zip DATA=~/<group>/data/Scoliosis1K-sil-pkl bash exp_2/setup_lab.sh
```
The dataset, checkpoints, logs and logits are not on GitHub.
