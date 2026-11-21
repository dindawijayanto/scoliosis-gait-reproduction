# checkpoint_2 / experiments

Full-scale training (lab GPU, Kaggle, RunPod), as scripts rather than notebooks. Each experiment has the same layout:

```
exp_N/
  README.md     question, variants, how to reproduce, caveats
  code/         every script of the experiment
  configs/      OpenGait configs that were trained
  partitions/   split files
  analysis/     results and the analysis of every run
  runs/         raw output (logits, logs, downloads), not in git
```

| experiment | question | status |
|---|---|---|
| `exp_1/` | Does choosing checkpoint and threshold on a val split fix the neutral failure? | never run; became `exp_2` A_base |
| `exp_2/` | Which imbalance fix survives a val-chosen threshold, do BAM / ordinal heads add anything, and how much do the models rely on the global body lean? | 8 runs, seed 0, frozen 2026-11-19. Start with `exp_2/analysis/runs.md` and `exp_2/analysis/STORY.md` |

The mini-data experiments that motivated these runs are in `../self_learning/` (notebooks 01-08).
