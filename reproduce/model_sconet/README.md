# model_sconet

OpenGait's code for ScoNet, copied from commit `4d228e1` with the code these configs never use removed (see `../README.md`), plus one added file.

| Config | Model class | Paper Table 4 row | Triplet labels |
|---|---|---|---|
| `configs/sconet.yaml` | `ScoNet_CE` | ScoNet | none (cross-entropy only) |
| `configs/sconet_triplet.yaml` | `ScoNet_Triplet` | ScoNet-Triplet | class (negative / neutral / positive) |
| `configs/sconet_mt.yaml` | `ScoNet` (official) | ScoNet-MT | subject ID |

- `opengait/modeling/models/sconet.py` is the official file. Its triplet loss uses `pids`, so the official config trains ScoNet-MT.
- `opengait/modeling/models/sconet_variants.py` is the only file I added. Both classes call the official `ScoNet.forward` and then change only the labels the triplet loss gets, or remove the triplet loss.
- `configs/official/sconet_scoliosis1k.yaml` is the untouched official config, kept for comparison. 

Run from this folder (needs Linux + NCCL, i.e. RunPod), or use `../runpod.py`:

```bash
torchrun --nproc_per_node=1 opengait/main.py --cfgs ./configs/sconet_mt.yaml --phase train --log_to_file
torchrun --nproc_per_node=1 opengait/main.py --cfgs ./configs/sconet_mt.yaml --phase test  --log_to_file
```
