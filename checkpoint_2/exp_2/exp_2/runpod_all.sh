#!/usr/bin/env bash
# Everything on one RunPod pod (2+ GPUs), from checkpoint_2/exp_2, inside tmux:
#   cd /workspace && git clone https://github.com/dindawijayanto/scoliosis-gait-reproduction.git
#   cd scoliosis-gait-reproduction/checkpoint_2/exp_2 && bash exp_2/runpod_all.sh
# Needs /workspace/Scoliosis1K-sil-pkl.zip. Steps: setup -> smoke test (100 it) -> full runs (16k) -> export -> calibrate
# -> /workspace/exp2_results.tgz (tables, logits, logs, 16k checkpoints). Download it, then TERMINATE the pod.
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-/workspace/data/Scoliosis1K-sil-pkl}
RUNS=${RUNS:-"D_la_s0 F_ord_on_D_la_s0 E_bam_on_D_la_s0 G_deskew_on_D_la_s0"}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
log() { echo "$(date '+%F %T') == $*"; }

log "1/5 setup"
SKIP_VENV=1 ZIP=${ZIP:-/workspace/Scoliosis1K-sil-pkl.zip} DATA=$DATA bash exp_2/setup_lab.sh > /workspace/setup.log 2>&1 \
  || { tail -20 /workspace/setup.log; exit 1; }
tail -4 /workspace/setup.log | head -1

export_parallel() {   # one export per GPU at a time
  local i=0 gpus=($GPUS)
  for r in $RUNS; do
    CUDA_VISIBLE_DEVICES=${gpus[$((i % ${#gpus[@]}))]} python exp_2/export_logits.py --runs "$r" $1 > "exp_2/logs/export_$r.log" 2>&1 &
    i=$((i + 1))
    if [ $((i % ${#gpus[@]})) -eq 0 ]; then wait; fi
  done
  wait
  grep -h "ALL DONE\|Error" exp_2/logs/export_*.log | sort | uniq -c
}

log "2/5 smoke test: every run 100 iterations, all at once"
rm -rf exp_2/logs exp_2/logits exp_2/results output/Scoliosis1K/ScoNetRobust
python exp_2/make_configs.py --data "$DATA" --iters 100 > /dev/null
sed -i "s/num_workers: 8/num_workers: $WORKERS/" exp_2/configs/*.yaml
GPUS="$GPUS" JOBS=2 bash exp_2/run_queue.sh $RUNS
export_parallel ""
python exp_2/calibrate.py > /workspace/smoke_summary.md
grep -c "^| " /workspace/smoke_summary.md > /dev/null && log "smoke test passed (table in /workspace/smoke_summary.md)"

log "3/5 full runs: 16000 iterations (= the 16k checkpoint of the author's 20k schedule)"
rm -rf exp_2/logs exp_2/logits exp_2/results output/Scoliosis1K/ScoNetRobust
python exp_2/make_configs.py --data "$DATA" > /dev/null
sed -i "s/total_iter: 20000/total_iter: 16000/; s/num_workers: 8/num_workers: $WORKERS/" exp_2/configs/*.yaml
GPUS="$GPUS" JOBS=$JOBS_PER_GPU bash exp_2/run_queue.sh $RUNS

log "4/5 export (all checkpoints + mirror / deskew probes + embeddings) and calibrate"
export_parallel ""
python exp_2/calibrate.py | tee /workspace/summary.md

log "5/5 pack results"
tar czf /workspace/exp2_results.tgz exp_2/results exp_2/logits exp_2/logs exp_2/configs \
  $(ls output/Scoliosis1K/ScoNetRobust/*/checkpoints/*-16000.pt)
ls -lh /workspace/exp2_results.tgz
log "DONE. Download /workspace/exp2_results.tgz, then TERMINATE the pod (stopping still bills storage)."
