#!/usr/bin/env bash
# One-time setup on the lab GPU server (Linux). Run from the OpenGait repo root:
#   DATA=/path/to/Scoliosis1K-sil-pkl bash exp_2/setup_lab.sh
# If DATA does not hold the 1493 subject folders yet, put Scoliosis1K-sil-pkl.zip next to the repo
# (or set ZIP=...) and the script unzips it there (it asks for the dataset password).
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-$(pwd)/datasets/Scoliosis1K/Scoliosis1K-sil-pkl}
ZIP=${ZIP:-$(pwd)/../Scoliosis1K-sil-pkl.zip}

echo "== 1/4 GPU"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo "== 2/4 Python packages (own venv inside the group folder: the account is shared)"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  [ -d ../venv ] || python3 -m venv ../venv
  source ../venv/bin/activate
  python -c "import torch" 2>/dev/null || pip install -q torch torchvision
fi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"
pip install -q tqdm pyyaml tensorboard opencv-python-headless kornia einops scikit-learn scipy matplotlib

echo "== 3/4 Data"
n=$(find "$DATA" -name '*.pkl' 2>/dev/null | wc -l)
if [ "$n" -ne 1493 ]; then
  [ -f "$ZIP" ] || { echo "no data at $DATA and no zip at $ZIP"; exit 1; }
  mkdir -p "$(dirname "$DATA")"
  unzip -q "$ZIP" -d "$(dirname "$DATA")"
  n=$(find "$DATA" -name '*.pkl' | wc -l)
fi
echo "pkl sequences: $n (expected 1493)"; [ "$n" -eq 1493 ]

echo "== 4/4 Configs"
grep -q "OPENGAIT_SEED" opengait/main.py || { echo "opengait/main.py has no OPENGAIT_SEED override: copy the exp_2 version of the repo"; exit 1; }
python exp_2/make_configs.py --data "$DATA"

cat <<MSG

Setup done. Every new shell: source ../venv/bin/activate
Inside tmux (so training survives a closed ssh session):
  tmux new -s exp2
  GPUS=0 JOBS=3 bash exp_2/run_queue.sh phase1   # pick a free GPU from nvidia-smi; JOBS = runs per GPU
Watch: tail -f exp_2/logs/<run>.log   |   nvidia-smi
After phase 1: python exp_2/export_logits.py && python exp_2/calibrate.py
Then phase 2 on the best imbalance fix, e.g.:
  python exp_2/make_configs.py --data "$DATA" --base B_strat && GPUS=0 bash exp_2/run_queue.sh phase2
MSG
