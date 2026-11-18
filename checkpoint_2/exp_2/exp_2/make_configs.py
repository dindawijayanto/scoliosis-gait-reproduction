"""Write one OpenGait config per (variant, seed) into exp_2/configs/.

Every variant starts from the author config (configs/sconet/sconet_scoliosis1k.yaml, the one that
reproduced ScoNet-MT 82.6 / 98.5 / 77.8) and changes only the parts named in VARIANTS.
All variants use the model ScoNetRobust and the loss LogitAdjustedCE; with tau=0, TripletSampler and
no aux head this is exactly ScoNet-MT (variant A).

Usage (from the OpenGait repo root):
  python exp_2/make_configs.py --data /path/to/Scoliosis1K-sil-pkl
  python exp_2/make_configs.py --data ... --base B_strat     # phase 2 heads on top of another imbalance fix
"""
import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
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
}
PHASE2 = ['E_bam', 'F_ord', 'G_deskew']


def build(author, variant, run, base, data_root, partition):
    cfg = copy.deepcopy(author)
    v = VARIANTS[variant]
    spec = dict(VARIANTS[base], **v) if variant in PHASE2 else v

    cfg['data_cfg'].update(dataset_root=data_root, dataset_partition=partition, num_workers=8, cache=True)
    cfg['model_cfg']['model'] = 'ScoNetRobust'
    cfg['model_cfg']['aux_head'] = spec.get('aux_head', 'none')

    ce = {'loss_term_weight': 1.0, 'scale': 16, 'tau': spec['tau'], 'type': 'LogitAdjustedCE',
          'log_prefix': 'softmax', 'log_accuracy': True}
    losses = [cfg['loss_cfg'][0], ce]                       # [triplet (unchanged), CE / LA-CE]
    aux_names = {'none': [], 'bam': ['bam'], 'ordinal': ['ord1', 'ord2']}[cfg['model_cfg']['aux_head']]
    for a in aux_names:                                     # aux heads: same loss, own binary prior
        losses.append(dict(ce, log_prefix=a, loss_term_weight=1.0 / len(aux_names), log_accuracy=False))
    cfg['loss_cfg'] = losses

    sampler = dict(spec['sampler'], batch_shuffle=True, batch_size=[8, 8], frames_num_fixed=30,
                   sample_type='fixed_unordered')
    cfg['trainer_cfg']['sampler'] = sampler
    tr = {'type': spec.get('transform', 'BaseSilCuttingTransform')}
    cfg['trainer_cfg']['transform'] = [tr]
    cfg['evaluator_cfg']['transform'] = [dict(tr)]
    cfg['evaluator_cfg']['sampler']['batch_size'] = 1       # = number of GPUs per run
    cfg['trainer_cfg'].update(save_iter=2000, save_name=run, find_unused_parameters=False)
    cfg['evaluator_cfg']['save_name'] = run
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder with the 1493 subject folders of Scoliosis1K-sil-pkl')
    ap.add_argument('--base', default='D_la', help='imbalance variant that phase 2 builds on')
    a = ap.parse_args()
    assert a.base in VARIANTS and a.base not in PHASE2
    author = yaml.safe_load(open(AUTHOR_CFG))
    partition = './exp_2/partitions/Scoliosis1K_118_trainval.json'   # runs start from the repo root
    out = HERE / 'configs'
    out.mkdir(exist_ok=True)
    for name, v in VARIANTS.items():
        for s in SEEDS:
            run = f'{name}_s{s}' if name not in PHASE2 else f'{name}_on_{a.base}_s{s}'
            cfg = build(author, name, run, a.base, a.data, partition)
            yaml.safe_dump(cfg, open(out / f'{run}.yaml', 'w'), sort_keys=False)
    print(f'wrote {len(list(out.glob("*.yaml")))} configs to {out}')


if __name__ == '__main__':
    main()
