"""Export per-child logits [N, 3, 16] for every checkpoint of every seed, on the val split and on the official test split.
Same inference path as OpenGait --phase test (all frames, float16). Output:
  <ws>/val_logits/<save_name>/{val,test}/iter_XXXXX.npz   (logits, true, subject)
Run from the OpenGait repo root on the pod: python /workspace/export_logits.py --ws /workspace
"""
import argparse, os, sys, tempfile, time
from pathlib import Path
import numpy as np

LABELS = ['negative', 'neutral', 'positive']
ap = argparse.ArgumentParser()
ap.add_argument('--ws', default='/workspace'); ap.add_argument('--seeds', default='0,1,2')
a = ap.parse_args()
REPO = Path.cwd(); assert (REPO / 'opengait' / 'main.py').exists(), 'run from the OpenGait repo root'
PARTS = {'val': REPO / 'datasets/Scoliosis1K/Scoliosis1K_118_trainval.json',
         'test': REPO / 'datasets/Scoliosis1K/Scoliosis1K_118_test.json'}

import torch, torch.distributed as dist
sys.path.insert(0, str(REPO / 'opengait'))
store = Path(tempfile.gettempdir()) / f'opengait_store_{os.getpid()}'; store.unlink(missing_ok=True)
dist.init_process_group('gloo', init_method=store.as_uri(), rank=0, world_size=1)
from utils import config_loader, get_msg_mgr
from modeling import models
get_msg_mgr().init_logger(str(Path(tempfile.gettempdir()) / 'opengait_export'), False)

for s in a.seeds.split(','):
    cfg_path = REPO / f'configs/sconet/sconet_mt_val_s{s}.yaml'
    base = config_loader(str(cfg_path)); name = base['trainer_cfg']['save_name']
    ckdir = REPO / 'output' / 'Scoliosis1K' / base['model_cfg']['model'] / name / 'checkpoints'
    for split, part in PARTS.items():
        cfgs = config_loader(str(cfg_path))
        cfgs['data_cfg'].update(dataset_partition=str(part), num_workers=0, cache=False)
        cfgs['evaluator_cfg']['restore_hint'] = 0
        m = getattr(models, cfgs['model_cfg']['model'])(cfgs, training=False)
        out_dir = Path(a.ws) / 'val_logits' / name / split; out_dir.mkdir(parents=True, exist_ok=True)
        for ck in sorted(ckdir.glob('*.pt')):
            it = int(ck.stem.split('-')[-1]); out = out_dir / f'iter_{it:05d}.npz'
            if out.exists(): continue
            m.load_state_dict(torch.load(ck, map_location='cuda:0', weights_only=False)['model']); m.eval()
            t = time.time()
            with torch.no_grad(): info = m.inference(0)
            logits = np.asarray(info['embeddings'], dtype=np.float32)
            true = np.array([LABELS.index(x.lower()) for x in m.test_loader.dataset.types_list])
            np.savez_compressed(out, logits=logits, true=true, subject=np.array(m.test_loader.dataset.label_list))
            print(f'{name} {split:4s} iter {it:5d}: n {len(true)} acc {100*(logits.mean(-1).argmax(-1)==true).mean():6.2f}  ({time.time()-t:.0f}s)', flush=True)
print('ALL DONE')
