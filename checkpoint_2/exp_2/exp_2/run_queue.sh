#!/usr/bin/env bash
# Train a list of exp_2 runs, JOBS at a time per GPU, skipping runs that already reached 20000 iterations.
# Run from the OpenGait repo root, inside tmux:
#   bash exp_2/run_queue.sh phase1                 # A, B, C, C_sqrt, D x 3 seeds (15 runs)
#   bash exp_2/run_queue.sh phase2                 # E, F, G on the --base used in make_configs (9 runs)
#   bash exp_2/run_queue.sh A_base_s0 D_la_s1      # or any config names
# Env (required on the shared lab server): GPUS="0" = which GPU(s) to use, check nvidia-smi for free ones first;
#      JOBS=3 = parallel runs per GPU (ScoNet needs ~4-6 GB each)
set -u
cd "$(dirname "$0")/.."
CFG_DIR=exp_2/configs
LOG_DIR=exp_2/logs; mkdir -p "$LOG_DIR"
[ -n "${GPUS:-}" ] || { echo "set GPUS, e.g. GPUS=0 (shared server: never take all GPUs). Current use:"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv; exit 1; }
JOBS=${JOBS:-3}

case "${1:-}" in
  phase1) RUNS=$(ls $CFG_DIR | sed 's/\.yaml$//' | grep -E '^(A_base|B_strat|C_up|C_sqrt|D_la)_s[0-9]$') ;;
  phase2) RUNS=$(ls $CFG_DIR | sed 's/\.yaml$//' | grep -E '^(E_bam|F_ord|G_deskew)_on_') ;;
  "") echo "usage: bash exp_2/run_queue.sh phase1|phase2|<run names>"; exit 1 ;;
  *) RUNS="$*" ;;
esac

done_already() {   # final checkpoint (total_iter of its config) exists?
  local it; it=$(grep -m1 "total_iter:" "$CFG_DIR/$1.yaml" | awk '{print $2}')
  ls output/Scoliosis1K/ScoNetRobust/"$1"/checkpoints/"$1"-$(printf %05d "$it").pt >/dev/null 2>&1
}

slots=(); for g in $GPUS; do for j in $(seq 1 "$JOBS"); do slots+=("$g"); done; done
echo "runs: $(echo $RUNS | wc -w) | GPUs: $GPUS | parallel slots: ${#slots[@]}"
port=29600; i=0; pids=()
for run in $RUNS; do
  if done_already "$run"; then echo "skip $run (already finished)"; continue; fi
  seed=${run##*_s}
  # wait for a free slot
  while [ "$(jobs -rp | wc -l)" -ge "${#slots[@]}" ]; do sleep 30; done
  gpu=${slots[$((i % ${#slots[@]}))]}; i=$((i + 1)); port=$((port + 1))
  echo "$(date '+%F %T') start $run  seed $seed  gpu $gpu  log $LOG_DIR/$run.log"
  OPENGAIT_SEED=$seed CUDA_VISIBLE_DEVICES=$gpu \
    torchrun --nproc_per_node=1 --master_port=$port opengait/main.py \
      --cfgs "$CFG_DIR/$run.yaml" --phase train --log_to_file > "$LOG_DIR/$run.log" 2>&1 &
  pid=$!
  sleep 90   # a broken setup crashes within seconds: stop the queue instead of starting the next run
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$(date '+%F %T') $run stopped in the first 90 s -> queue stopped. Error:"
    grep -m 3 -E "Error|error:" "$LOG_DIR/$run.log"; exit 1
  fi
done
wait
echo "$(date '+%F %T') all runs finished"
