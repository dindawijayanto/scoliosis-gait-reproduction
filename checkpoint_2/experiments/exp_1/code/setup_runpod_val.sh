#!/usr/bin/env bash
# ScoNet-MT on the reduced train split (train minus a held-out val set), 3 seeds.
# Upload to /workspace: this file, Scoliosis1K-sil-pkl.zip, Scoliosis1K_118_trainval.json,
# Scoliosis1K_118_test.json, export_logits.py. Then: bash /workspace/setup_runpod_val.sh
# Config = the author config used for the 16k reproduction; only partition, save_name and seed differ.
set -e
WS=${WS:-/workspace}
cd "$WS"

echo "== 1/5 GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== 2/5 OpenGait code"
[ -d OpenGait ] || git clone https://github.com/ShiqiYu/OpenGait.git
pip install -q tqdm pyyaml tensorboard opencv-python-headless kornia einops scikit-learn imageio matplotlib fvcore lightly transformers
# OpenGait seeds with the DDP rank only (always 0 on one GPU). Add an env override: SEED=k.
cd "$WS/OpenGait"
grep -q "OPENGAIT_SEED" opengait/main.py || \
  sed -i "s#seed = torch.distributed.get_rank()#seed = torch.distributed.get_rank() + int(os.environ.get('OPENGAIT_SEED', 0))#" opengait/main.py
grep -n "OPENGAIT_SEED" opengait/main.py

echo "== 3/5 Data"
if [ "$(find "$WS/data/Scoliosis1K-sil-pkl" -name '*.pkl' 2>/dev/null | wc -l)" -ne 1493 ]; then
  command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip; }
  rm -rf "$WS/data/Scoliosis1K-sil-pkl" && mkdir -p "$WS/data"
  unzip -q "$WS/Scoliosis1K-sil-pkl.zip" -d "$WS/data"     # asks for the dataset password
fi
echo "pkl sequences found: $(find "$WS/data/Scoliosis1K-sil-pkl" -name '*.pkl' | wc -l)  (expected 1493)"
cp "$WS"/Scoliosis1K_118_trainval.json "$WS"/Scoliosis1K_118_test.json datasets/Scoliosis1K/

echo "== 4/5 Configs (one per seed)"
for S in 0 1 2; do
  CFG=configs/sconet/sconet_mt_val_s$S.yaml
  sed -e "s#dataset_root: .*#dataset_root: $WS/data/Scoliosis1K-sil-pkl#" \
      -e "s#Scoliosis1K_\*\*\*.json#Scoliosis1K_118_trainval.json#" \
      -e "s#num_workers: 1#num_workers: 8\n  cache: true#" \
      -e "s#save_iter: 20000#save_iter: 2000#" \
      -e "s#batch_size: 8#batch_size: 1#" \
      -e "s#save_name: ScoNet#save_name: ScoNet_MT_val_s$S#" \
      configs/sconet/sconet_scoliosis1k.yaml > "$CFG"
  echo "--- $CFG"; diff configs/sconet/sconet_scoliosis1k.yaml "$CFG" || true
done

echo "== 5/5 PyTorch"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"

cat <<MSG

Setup done. Train each seed in its own tmux window (they fit together on one 48 GB GPU; use different ports):
  tmux new -s s0   then: cd $WS/OpenGait && OPENGAIT_SEED=0 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 opengait/main.py --cfgs ./configs/sconet/sconet_mt_val_s0.yaml --phase train --log_to_file
  tmux new -s s1   then: same with OPENGAIT_SEED=1, --master_port=29501, sconet_mt_val_s1.yaml
  tmux new -s s2   then: same with OPENGAIT_SEED=2, --master_port=29502, sconet_mt_val_s2.yaml
After all three finish, export val + test logits for every checkpoint:
  cd $WS/OpenGait && python $WS/export_logits.py --ws $WS
Download $WS/val_logits/ back to checkpoint_2/experiments/exp_1/runs/val_logits/ and run calibrate.py locally.
MSG
