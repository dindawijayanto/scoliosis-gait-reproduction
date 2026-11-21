#!/usr/bin/env bash
# exp_2 on Kaggle (GPU T4 x2), straight from a git clone of this repo: B_strat seeds 1 and 2, one run per GPU in
# parallel (~7.5 h; B_strat_s0 took 7 h 20 min on Kaggle). The configs are B_strat_s0 with only save_name changed;
# the seed comes from OPENGAIT_SEED (opengait/main.py). Each run's val/test logits (+ mirror / deskew probes) are
# exported right after it. Output: /kaggle/working/exp2_kaggle_Bseeds.tgz -> runs/incoming/kaggle/ on the laptop.
#   !git clone --depth 1 -b <branch> https://github.com/<user>/<repo>.git /kaggle/working/OpenGait
#   !bash /kaggle/working/OpenGait/checkpoint_2/experiments/exp_2/code/kaggle_Bseeds.sh
set -e
REPO=$(cd "$(dirname "$0")/../../../.." && pwd); cd "$REPO"
EXP=checkpoint_2/experiments/exp_2
DATA=${DATA:-/kaggle/input/datasets/dindawijayanto/scoliosis-1k/Scoliosis1K-sil-pkl}
RUNS=${RUNS:-"B_strat_s1 B_strat_s2"}
NGPU=$(nvidia-smi -L | wc -l)
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "== packages"
pip install -q tqdm pyyaml tensorboard opencv-python-headless kornia einops scikit-learn scipy imageio matplotlib fvcore lightly transformers 2>&1 | tail -2 || true

echo "== data"
if [ ! -d "$DATA" ]; then DATA=$(find /kaggle/input -maxdepth 6 -type d -name Scoliosis1K-sil-pkl | head -1); fi
n=$(find "$DATA" -name '*.pkl' | wc -l); echo "data: $DATA  pkl sequences: $n (expected 1493)"; [ "$n" -eq 1493 ]
for r in $RUNS; do sed -i "s#dataset_root: .*#dataset_root: $DATA#" $EXP/configs/$r.yaml; done
mkdir -p $EXP/runs/logs

worker() {         # $1 = GPU index, rest = runs for this GPU
  local gpu=$1; shift
  local port=$((29700 + gpu * 10))
  for r in "$@"; do
    port=$((port + 1))
    local seed=${r##*_s}
    echo "$(date '+%T') gpu $gpu: train $r (seed $seed)"
    OPENGAIT_SEED=$seed CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$port opengait/main.py \
      --cfgs $EXP/configs/$r.yaml --phase train --log_to_file > $EXP/runs/logs/$r.log 2>&1 \
      || { echo "$(date '+%T') gpu $gpu: $r FAILED"; tail -5 $EXP/runs/logs/$r.log; continue; }
    echo "$(date '+%T') gpu $gpu: export $r"
    CUDA_VISIBLE_DEVICES=$gpu python $EXP/code/export_logits.py --runs $r --iters 12000,14000,16000 \
      > $EXP/runs/logs/export_$r.log 2>&1 || echo "$(date '+%T') gpu $gpu: export of $r FAILED (see runs/logs/export_$r.log)"
    echo "$(date '+%T') gpu $gpu: $r done"
  done
}

echo "== train + export on $NGPU GPU(s)"
pids=(); g=0
if [ "$NGPU" -ge 2 ]; then
  for r in $RUNS; do worker $((g % NGPU)) $r & pids+=($!); g=$((g + 1)); sleep 20; done
else
  worker 0 $RUNS & pids+=($!)          # one GPU: the runs go one after the other (~15 h, too long for one session)
fi
for p in "${pids[@]}"; do wait $p; done

echo "== pack"
cks=""
for r in $RUNS; do
  f=$(ls output/Scoliosis1K/ScoNetRobust/$r/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  [ -n "$f" ] && cks="$cks $f"
done
logits=""; for r in $RUNS; do [ -d $EXP/runs/logits/$r ] && logits="$logits $EXP/runs/logits/$r"; done
tar czf /kaggle/working/exp2_kaggle_Bseeds.tgz $logits $EXP/runs/logs $cks
ls -la /kaggle/working/exp2_kaggle_Bseeds.tgz
echo "$(date '+%T') DONE -> /kaggle/working/exp2_kaggle_Bseeds.tgz"
