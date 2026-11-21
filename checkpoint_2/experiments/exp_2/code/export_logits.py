"""Export everything one exp_2 run can tell, not only its accuracy.

For every checkpoint of every run (learning curves, val-chosen checkpoint):
  checkpoint_2/experiments/exp_2/runs/logits/<run>/{val,test}/iter_XXXXX.npz   logits [N,3,p], aux [N,T,2,p] (runs with a head), true, subject
For the analysis checkpoint (16000, or the last one of a short run) also:
  checkpoint_2/experiments/exp_2/runs/logits/<run>/test/iter_XXXXX.npz         + embed [N,256,16] (separability, t-SNE)
  checkpoint_2/experiments/exp_2/runs/logits/<run>/test_mirror/iter_XXXXX.npz  test children mirrored left/right   (lean shortcut probe)
  checkpoint_2/experiments/exp_2/runs/logits/<run>/test_deskew/iter_XXXXX.npz  test children with the global lean removed (lean-corrected test)
  checkpoint_2/experiments/exp_2/runs/logits/<run>/test_shuffle/iter_XXXXX.npz test frames in random order (only runs trained on ordered frames)
Same inference path as OpenGait --phase test (all frames, float16).
Run from anywhere: python checkpoint_2/experiments/exp_2/code/export_logits.py [--runs A_base_s0,D_la_s1] [--iters 16000] [--no-probes]
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

LABELS = ['negative', 'neutral', 'positive']
HERE = Path(__file__).resolve().parent            # exp_2/code
EXP = HERE.parent                                   # checkpoint_2/experiments/exp_2
REPO = EXP.parents[2]                               # OpenGait
PARTS = {'val': EXP / 'partitions' / 'Scoliosis1K_118_trainval.json',
         'test': EXP / 'partitions' / 'Scoliosis1K_118_test.json'}

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='', help='comma list; default = every config with checkpoints')
ap.add_argument('--iters', default='', help='comma list; default = every saved checkpoint')
ap.add_argument('--no-probes', action='store_true', help='skip mirror / deskew / embeddings')
a = ap.parse_args()
os.chdir(REPO)

import torch
import torch.distributed as dist
sys.path.insert(0, str(REPO / 'opengait'))
if not dist.is_initialized():
    store = Path(tempfile.gettempdir()) / f'opengait_store_{os.getpid()}'
    store.unlink(missing_ok=True)
    dist.init_process_group('gloo', init_method=store.as_uri(), rank=0, world_size=1)
from utils import config_loader, get_msg_mgr
from modeling import models
get_msg_mgr().init_logger(str(Path(tempfile.gettempdir()) / 'opengait_export'), False)


def probe_transform(cfgs, probe):
    deskewed = cfgs['evaluator_cfg']['transform'][0]['type'] == 'DeskewSilCuttingTransform'
    if probe == 'mirror':
        return [{'type': 'MirrorSilCuttingTransform', 'deskew': deskewed}]
    if probe == 'shuffle':   # frame order destroyed at test (order-aware runs only)
        return [{'type': 'ShuffleSilCuttingTransform', 'deskew': deskewed}]
    return [{'type': 'DeskewSilCuttingTransform'}]


def probes_of(run):
    cfgs = config_loader(str(EXP / 'configs' / f'{run}.yaml'))
    ordered = cfgs['trainer_cfg']['sampler']['sample_type'].split('_')[1] in ('ordered', 'allordered')   # not 'unordered'
    return ('mirror', 'deskew', 'shuffle') if ordered else ('mirror', 'deskew')


def build(run, part, probe=None):
    cfgs = config_loader(str(EXP / 'configs' / f'{run}.yaml'))
    cfgs['data_cfg'].update(dataset_partition=str(part), num_workers=0, cache=False)
    cfgs['evaluator_cfg']['restore_hint'] = 0
    if probe is not None:
        cfgs['evaluator_cfg']['transform'] = probe_transform(cfgs, probe)
    return getattr(models, cfgs['model_cfg']['model'])(cfgs, training=False)


def export(m, ck, out, keep_embed):
    m.load_state_dict(torch.load(ck, map_location='cuda:0', weights_only=False)['model'])
    m.eval()
    t = time.time()
    with torch.no_grad():
        info = m.inference(0)
    logits = np.asarray(info['embeddings'], dtype=np.float32)
    extra = {}
    if 'aux' in info:
        extra['aux'] = np.asarray(info['aux'], dtype=np.float32)
    if keep_embed and 'embed' in info:
        extra['embed'] = np.asarray(info['embed'], dtype=np.float16)
    true = np.array([LABELS.index(x.lower()) for x in m.test_loader.dataset.types_list])
    np.savez_compressed(out, logits=logits, true=true, subject=np.array(m.test_loader.dataset.label_list), **extra)
    return 100 * (logits.mean(-1).argmax(-1) == true).mean(), len(true), time.time() - t


runs = a.runs.split(',') if a.runs else sorted(p.stem for p in (EXP / 'configs').glob('*.yaml'))
want = {int(x) for x in a.iters.split(',')} if a.iters else None
for run in runs:
    ckdir = REPO / 'output' / 'Scoliosis1K' / 'ScoNetRobust' / run / 'checkpoints'
    cks = {int(c.stem.split('-')[-1]): c for c in ckdir.glob('*.pt')}
    cks = {it: c for it, c in cks.items() if not want or it in want}
    if not cks:
        continue
    probe_it = 16000 if 16000 in cks else max(cks)
    jobs = [(split, part, None, list(cks)) for split, part in PARTS.items()]
    if not a.no_probes:
        jobs += [(f'test_{p}', PARTS['test'], p, [probe_it]) for p in probes_of(run)]
    for split, part, probe, iters in jobs:
        out_dir = EXP / 'runs' / 'logits' / run / split
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [it for it in sorted(iters) if not (out_dir / f'iter_{it:05d}.npz').exists()]
        if not todo:
            continue
        m = build(run, part, probe)
        for it in todo:
            keep = split == 'test' and it == probe_it and not a.no_probes
            acc, n, sec = export(m, cks[it], out_dir / f'iter_{it:05d}.npz', keep)
            print(f'{run} {split:11s} iter {it:5d}: n {n} acc {acc:6.2f}  ({sec:.0f}s)', flush=True)
        del m
        torch.cuda.empty_cache()
print('ALL DONE')
