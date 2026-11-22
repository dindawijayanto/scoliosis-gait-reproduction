"""Write one OpenGait config per (variant, seed) into checkpoint_2/experiments/exp_2/configs/.

Every variant starts from the author config (configs/sconet/sconet_scoliosis1k.yaml, the one that
reproduced ScoNet-MT 82.6 / 98.5 / 77.8) and changes only the parts named in VARIANTS.
All variants use the model ScoNetRobust and the loss LogitAdjustedCE; with tau=0, TripletSampler and
no aux head this is exactly ScoNet-MT (variant A).

Usage (from the OpenGait repo root):
  python checkpoint_2/experiments/exp_2/code/make_configs.py --data /path/to/Scoliosis1K-sil-pkl
  python checkpoint_2/experiments/exp_2/code/make_configs.py --data ... --base B_strat     # phase 2 heads on top of another imbalance fix
"""
import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent            # exp_2/code
EXP = HERE.parent                                   # checkpoint_2/experiments/exp_2
REPO = EXP.parents[2]                               # OpenGait
AUTHOR_CFG = REPO / 'configs' / 'sconet' / 'sconet_scoliosis1k.yaml'
SEEDS = [0, 1, 2]

TRIPLET = {'type': 'TripletSampler'}
VARIANTS = {
    # phase 1: how to fix the class imbalance (the 1:1:8 train set)
    'A_base':   {'sampler': TRIPLET, 'tau': 0.0},                                            # ScoNet-MT
    'B_strat':  {'sampler': {'type': 'ClassAwareSampler', 'mode': 'stratified'}, 'tau': 0.0},  # every batch ~3/3/2
    'C_up':     {'sampler': {'type': 'ClassAwareSampler', 'mode': 'upsample', 'power': 1.0}, 'tau': 0.0},
    'C_sqrt':   {'sampler': {'type': 'ClassAwareSampler', 'mode': 'upsample', 'power': 0.5}, 'tau': 1.0},
    'D_la':     {'sampler': TRIPLET, 'tau': 1.0},                                            # LA loss
    # phase 2: architecture parts on top of the imbalance fix given by --base (default D_la)
    'E_bam':    {'aux_head': 'bam'},
    'F_ord':    {'aux_head': 'ordinal'},
    'G_deskew': {'transform': 'DeskewSilCuttingTransform'},
    # phase 2b: the other checkpoint_2 notebook ideas, checked at full scale
    'H_tattn':  {'model': {'temporal': 'attention'}},          # notebook 04: gated frame attention
    'I_strip':  {'model': {'strip_weights': 'dynamic'}},       # notebook 03: learned per-child strip weights
    'J_ctri':   {'model': {'class_triplet': True}},            # notebook 07: triplet with class as identity
    'K_gei':    {'model': {'gei_channel': True}},              # notebook 05: ScoliGaitX GEI-subtract, early fusion
    'L_flip':   {'model': {'train_flip': True}},               # notebook 07/08: flip (expected to hurt)
}
PHASE2 = ['E_bam', 'F_ord', 'G_deskew', 'H_tattn', 'I_strip', 'J_ctri', 'K_gei', 'L_flip']


def build(author, variant, run, base, data_root, partition, iters=20000):
    cfg = copy.deepcopy(author)
    v = VARIANTS[variant]
    spec = dict(VARIANTS[base], **v) if variant in PHASE2 else v

    cfg['data_cfg'].update(dataset_root=data_root, dataset_partition=partition, num_workers=8, cache=True)
    cfg['model_cfg']['model'] = 'ScoNetRobust'
    cfg['model_cfg']['aux_head'] = spec.get('aux_head', 'none')
    cfg['model_cfg'].update(spec.get('model', {}))
    if cfg['model_cfg'].get('gei_channel'):
        cfg['model_cfg']['backbone_cfg']['in_channel'] = 2

    ce = {'loss_term_weight': 1.0, 'scale': 16, 'tau': spec['tau'], 'type': 'LogitAdjustedCE',
          'log_prefix': 'softmax', 'log_accuracy': True}
    losses = [cfg['loss_cfg'][0], ce]                       # [triplet (unchanged), CE / LA-CE]
    aux_names = {'none': [], 'bam': ['bam'], 'ordinal': ['ord1', 'ord2']}[cfg['model_cfg']['aux_head']]
    for a in aux_names:                                     # aux heads: same loss, own binary prior
        losses.append(dict(ce, log_prefix=a, loss_term_weight=1.0 / len(aux_names), log_accuracy=False))
    if cfg['model_cfg'].get('strip_weights', 'none') != 'none':        # the strip-weighted decision gets its own CE
        losses.append(dict(ce, log_prefix='strip', log_accuracy=False))
    if cfg['model_cfg'].get('class_triplet'):
        losses.append({'loss_term_weight': 1.0, 'margin': 0.2, 'type': 'ClassTripletLoss', 'log_prefix': 'class_triplet'})
    cfg['loss_cfg'] = losses

    sampler = dict(spec['sampler'], batch_shuffle=True, batch_size=[8, 8], frames_num_fixed=30,
                   sample_type='fixed_unordered')
    cfg['trainer_cfg']['sampler'] = sampler
    tr = {'type': spec.get('transform', 'BaseSilCuttingTransform')}
    cfg['trainer_cfg']['transform'] = [tr]
    cfg['evaluator_cfg']['transform'] = [dict(tr)]
    cfg['evaluator_cfg']['sampler']['batch_size'] = 1       # = number of GPUs per run
    # short protocol: same schedule shape as the author's (lr drops at 50/70/90% of training)
    cfg['trainer_cfg'].update(total_iter=iters, save_iter=2000 if iters == 20000 else iters // 2,
                              save_name=run, find_unused_parameters=False)
    cfg['scheduler_cfg']['milestones'] = [int(iters * f) for f in (0.5, 0.7, 0.9)]
    cfg['evaluator_cfg']['restore_hint'] = iters
    cfg['evaluator_cfg']['save_name'] = run
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder with the 1493 subject folders of Scoliosis1K-sil-pkl')
    ap.add_argument('--base', default='D_la', help='imbalance variant that phase 2 builds on')
    ap.add_argument('--iters', type=int, default=20000, help='total iterations (author: 20000); milestones scale with it')
    a = ap.parse_args()
    assert a.base in VARIANTS and a.base not in PHASE2
    author = yaml.safe_load(open(AUTHOR_CFG))
    partition = './checkpoint_2/experiments/exp_2/partitions/Scoliosis1K_118_trainval.json'   # runs start from the repo root
    out = EXP / 'configs'
    out.mkdir(exist_ok=True)
    for name, v in VARIANTS.items():
        for s in SEEDS:
            run = f'{name}_s{s}' if name not in PHASE2 else f'{name}_on_{a.base}_s{s}'
            cfg = build(author, name, run, a.base, a.data, partition, a.iters)
            yaml.safe_dump(cfg, open(out / f'{run}.yaml', 'w'), sort_keys=False)
    print(f'wrote {len(list(out.glob("*.yaml")))} configs to {out}')


if __name__ == '__main__':
    main()
