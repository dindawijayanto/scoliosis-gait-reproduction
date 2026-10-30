# model_cnn

The CNN baselines of paper Table 4, run with the same OpenGait code as `model_sconet` (commit `4d228e1`, with unused code removed; see `../README.md`).

| Config | Model class | Paper Table 4 row | Triplet labels |
|---|---|---|---|
| `configs/cnn.yaml` | `CNN_CE` | CNN | none (cross-entropy only) |
| `configs/cnn_triplet.yaml` | `CNN_Triplet` | CNN-Triplet | class |
| `configs/cnn_mt.yaml` | `CNN_MT` | CNN-MT | subject ID |


```
ResNet9 per frame -> temporal max   [n, 512, 16, 11]    same as ScoNet
global pool (mean + max)            [n, 512]
fc Linear(512, 256, no bias)        [n, 256]  embedding (triplet loss)
classifier Linear(256, 3, no bias)  [n, 3]    logits (cross-entropy)
```

All training settings are the paper's, identical to the ScoNet configs. The one difference is the cross-entropy `scale`: 16 for ScoNet, whose logits are cosines in [−1, 1], and 1 here, where the logits are plain dot products. 5.03 M parameters (ScoNet 7.01 M).

Run from this folder on RunPod, or use `../runpod.py`:

```bash
torchrun --nproc_per_node=1 opengait/main.py --cfgs ./configs/cnn_mt.yaml --phase train --log_to_file
```
