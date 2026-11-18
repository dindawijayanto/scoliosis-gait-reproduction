"""Export per-child logits for every checkpoint of every exp_2 run, on the val split and the official test split.

Same inference path as OpenGait --phase test (all frames, float16). Output:
  exp_2/logits/<run>/{val,test}/iter_XXXXX.npz   (logits [N,3,16], aux [N,T,2,16] if the run has a head, true, subject)
Run from the OpenGait repo root: python exp_2/export_logits.py [--runs A_base_s0,D_la_s1] [--iters 16000,20000]
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

LABELS = ['negative', 'neutral', 'positive']
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PARTS = {'val': HERE / 'partitions' / 'Scoliosis1K_118_trainval.json',
         'test': HERE / 'partitions' / 'Scoliosis1K_118_test.json'}

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='', help='comma list; default = every config with checkpoints')
ap.add_argument('--iters', default='', help='comma list; default = every saved checkpoint')
a = ap.parse_args()
os.chdir(REPO)

import torch
import torch.distributed as dist
sys.path.insert(0, str(REPO / 'opengait'))
store = Path(tempfile.gettempdir()) / f'opengait_store_{os.getpid()}'
store.unlink(missing_ok=True)
dist.init_process_group('gloo', init_method=store.as_uri(), rank=0, world_size=1)
from utils import config_loader, get_msg_mgr
from modeling import models
get_msg_mgr().init_logger(str(Path(tempfile.gettempdir()) / 'opengait_export'), False)

runs = a.runs.split(',') if a.runs else sorted(p.stem for p in (HERE / 'configs').glob('*.yaml'))
want = {int(x) for x in a.iters.split(',')} if a.iters else None
for run in runs:
    ckdir = REPO / 'output' / 'Scoliosis1K' / 'ScoNetRobust' / run / 'checkpoints'
    cks = sorted(ckdir.glob('*.pt'))
    if not cks:
        continue
    for split, part in PARTS.items():
        cfgs = config_loader(str(HERE / 'configs' / f'{run}.yaml'))
        cfgs['data_cfg'].update(dataset_partition=str(part), num_workers=0, cache=False)
        cfgs['evaluator_cfg']['restore_hint'] = 0
        m = getattr(models, cfgs['model_cfg']['model'])(cfgs, training=False)
        out_dir = HERE / 'logits' / run / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for ck in cks:
            it = int(ck.stem.split('-')[-1])
            out = out_dir / f'iter_{it:05d}.npz'
            if out.exists() or (want and it not in want):
                continue
            m.load_state_dict(torch.load(ck, map_location='cuda:0', weights_only=False)['model'])
            m.eval()
            t = time.time()
            with torch.no_grad():
                info = m.inference(0)
            logits = np.asarray(info['embeddings'], dtype=np.float32)
            extra = {'aux': np.asarray(info['aux'], dtype=np.float32)} if 'aux' in info else {}
            true = np.array([LABELS.index(x.lower()) for x in m.test_loader.dataset.types_list])
            np.savez_compressed(out, logits=logits, true=true,
                                subject=np.array(m.test_loader.dataset.label_list), **extra)
            acc = 100 * (logits.mean(-1).argmax(-1) == true).mean()
            print(f'{run} {split:4s} iter {it:5d}: n {len(true)} acc {acc:6.2f}  ({time.time() - t:.0f}s)', flush=True)
print('ALL DONE')
