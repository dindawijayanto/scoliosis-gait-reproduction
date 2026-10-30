# Reproduction of ScoNet and the CNN baselines (paper Table 4)

```
reproduce/
  runpod.py            one command per step on RunPod (setup, train, test, sweep, pack)
  requirements.txt
  datasets/            Scoliosis1K pkl files (local only) + the four split files
  model_sconet/        official OpenGait code + ScoNet, ScoNet-Triplet, ScoNet-MT configs
  model_cnn/           same OpenGait code + the CNN head + CNN, CNN-Triplet, CNN-MT configs
  result_sconet/       downloaded ScoNet runs + analysis_sconet.ipynb
  result_cnn/          downloaded CNN runs + analysis_cnn.ipynb
  analysis/            compare_with_paper.ipynb (all runs vs Table 4)
```

## What is official and what I added

**Copied from OpenGait (commit 4d228e1), then trimmed:** everything under `model_*/opengait/` except the two files below, `configs/default.yaml`, `configs/official/sconet_scoliosis1k.yaml`, and the split files in `datasets/`. Only the files ScoNet needs were copied; `models/__init__.py` imports every file in its folder, so leaving out the other models means their extra packages are not needed.

Code the six configs never reach was then removed: `evaluation/metric.py` and `evaluation/re_rank.py`; every eval function except `evaluate_scoliosis`; every transform except `BaseSilCuttingTransform`; every sampler except `TripletSampler` (train) and `InferenceSampler` (test, from `default.yaml`); the point-cloud, D-Gait and self-supervised branches in `collate_fn.py` and `dataset.py`; and the unused layers in `modules.py` (`model_cnn` keeps only `SetBlockWrapper`, `PackSequenceWrapper`, `BasicConv2d`). The model files, `base_model.py`, the losses and ResNet9 are unchanged, and a check on real data gave the same batches, logits, losses and parameter counts before and after.

**Added, because the repo does not ship them:**

| File | Why |
|---|---|
| `model_sconet/opengait/modeling/models/sconet_variants.py` | `ScoNet_CE` and `ScoNet_Triplet`. They subclass the official `ScoNet` and change only which loss gets which labels. |
| `model_cnn/opengait/modeling/models/cnn.py` | `CNN_CE`, `CNN_Triplet`, `CNN_MT`. Same ResNet9 and temporal max as ScoNet; the head is global pool → Linear 512→256 → Linear 256→3, as in `self_learning/cnn`. The paper does not publish its CNN code. |
| `model_*/configs/*.yaml` | One per variant, generated from the official config. Each file lists the lines that differ at the top. |

**Paper settings, the same in all six configs:** ResNet9 (64-128-256-512, strides 1-2-2-1, no maxpool), input 64×44, 30 frames per clip, batch 8 children × 8 clips, SGD lr 0.1 / momentum 0.9 / weight decay 5e-4, lr ×0.1 at 10k, 14k, 18k, 20,000 iterations, float16, triplet margin 0.2, cross-entropy with label smoothing 0.1, split 1:1:8.

**Lines changed from the official config, and why:**

| Line | Official | Here | Why |
|---|---|---|---|
| `dataset_root`, `dataset_partition` | placeholders | `../datasets/...`, `Scoliosis1K_118.json` | the official file leaves them for the user; 1:1:8 is the split of Table 4 |
| `num_workers`, `cache` | 1, false | 8, true | loading speed only |
| evaluator `batch_size` | 8 | 1 | OpenGait requires test batch = number of GPUs (`base_model.py:442`); I use 1 GPU. Predictions do not depend on it. |
| `save_iter` | 20000 | 2000 | keep a checkpoint every 2k so each can be tested |
| `save_name` | ScoNet | one per variant | separate output folders |
| CE `scale` (CNN only) | 16 | 1 | ScoNet's logits are cosines in [−1, 1] and need ×16; the CNN's are plain dot products |

## Running a variant on RunPod

1. Create a pod with **1× A40** (48 GB) and the PyTorch template. One run takes about 2 h 50 min.
2. On the laptop, run `python runpod.py bundle`. It writes `reproduce_code.zip` (code and split files only, under 1 MB). Upload it and `datasets/Scoliosis1K-sil-pkl.zip` (96 MB) to `/workspace` on the pod. In the Jupyter terminal:
   ```bash
   cd /workspace && unzip -q reproduce_code.zip && cd reproduce
   mv /workspace/Scoliosis1K-sil-pkl.zip datasets/
   python runpod.py setup                              # asks for the dataset password
   tmux new -s run
   python runpod.py all sconet_triplet                 # train, test every checkpoint, zip
   ```
   Detach from tmux with Ctrl+B then D; reattach with `tmux attach -t run`.
3. Download `results_<SaveName>.zip` from the file browser, then **terminate the pod**.
4. On the laptop, extract it into `result_sconet/<SaveName>/` or `result_cnn/<SaveName>/` and run that folder's notebook, then `analysis/compare_with_paper.ipynb`.

Variant names for `runpod.py`: `sconet_mt`, `sconet_triplet`, `sconet`, `cnn_mt`, `cnn_triplet`, `cnn`.
