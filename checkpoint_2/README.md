# checkpoint_2

Checkpoint 1 reproduced ScoNet-MT (82.6 / 98.5 / 77.8). Checkpoint 2 asks why it almost never recognises the neutral class, whether the competitors' gains are real, and what the model actually reads from the silhouettes.

```
checkpoint_2/
  self_learning/     learning on small data (laptop): one notebook per topic, mini models
    01_class_imbalance ... 08_flip_asymmetry
    tools/common.py  shared data, model, training and metric code
  experiments/       full-scale training: code, configs and the analysis of every run
    exp_1/           val calibration (never run, became exp_2 A_base)
    exp_2/           8 full runs: imbalance fixes, BAM / ordinal heads, deskew, flip
```

Read in this order:

1. `self_learning/README.md` and notebooks 01-08: the diagnosis and the ideas, tested on mini models.
2. `experiments/exp_2/README.md`: the full-scale protocol and variants.
3. `experiments/exp_2/analysis/runs.md`: what every training run gave.
4. `experiments/exp_2/analysis/STORY.md`: the findings in presentation order.

Main findings (exp_2, seed 0):
- A threshold chosen on val closes most of the gap: ScoNet-MT goes from 76.0 to 87.3 balanced accuracy without retraining.
- Only the class-stratified sampler gives a significant gain after calibration (89.7, +2.3 [+0.2, +4.5]).
- Every model reads the global body lean: mirroring the input inverts the ranking (AUC 97 -> 27). Deskewed training halves this (mirror AUC 59) for about -2 points.
