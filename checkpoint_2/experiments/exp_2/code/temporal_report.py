"""exp_2 phase 3 report: do order-aware temporal models help? -> analysis/temporal.md

Runs (any subset that has logits):
  T_hppmax_on_B_strat_s{0,1,2}  control: fine-tune of B_strat_s0@16k on ordered clips, per-frame HPP then max (order-free)
  T_tconv_on_B_strat_s{0,1,2}   + temporal conv over the ordered frames
  T_pose_on_B_strat_s{0,1,2}    + temporal conv + mirror-invariant posture stream
  S_tconv_strat_s0, S_pose_strat_s0   the same two models trained from scratch (16k), compared with B_strat_s0
  ALL_strat_tconv_la_s{0,1,2}         S_tconv + LA loss on the train prior (80/10/10), from scratch, 3 seeds

Per run: checkpoint + tau chosen on val (calibrate.py rules), test metrics, and at the probe checkpoint (last one)
the mirrored / deskewed / shuffled-frame test. The shuffle probe keeps the val tau: if a model uses frame order,
its test score drops on shuffled frames; an order-free model gives exactly the same numbers.

python checkpoint_2/experiments/exp_2/code/temporal_report.py [--tgz path/to/exp2_runpod_T.tgz] [--no-pav]
"""
import argparse
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from calibrate import LOGP, evaluate_run, metrics, threshold_free, paired_bootstrap   # noqa: E402

LOGITS = EXP / 'runs' / 'logits'


def merge(tgz):
    """Unpack a RunPod archive into runs/incoming/runpod/<name>/ and copy its logits into runs/logits/."""
    tgz = Path(tgz)
    dest = EXP / 'runs' / 'incoming' / 'runpod' / tgz.name.split('.')[0]
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as t:
        members = [m for m in t.getmembers() if not (m.name.startswith('/') or '..' in Path(m.name).parts)]
        t.extractall(dest, members=members)
    n = 0
    for run in sorted((dest / 'exp_2' / 'logits').iterdir()):
        for split in run.iterdir():
            out = LOGITS / run.name / split.name
            out.mkdir(parents=True, exist_ok=True)
            for f in split.glob('*.npz'):
                shutil.copy2(f, out / f.name); n += 1
    print(f'merged {n} logit files from {tgz.name} into {LOGITS}')


def probe(run_dir, rows):
    """Metrics on the probe test sets at the last checkpoint, with the val-tau of the run."""
    it = rows['fixed']['iter']
    tau = rows['val-tau']['tau']
    out = {}
    for split in ('test', 'test_mirror', 'test_deskew', 'test_shuffle'):
        f = run_dir / split / f'iter_{it:05d}.npz'
        if f.exists():
            d = np.load(f)
            o = np.argsort(d['subject'])
            z, y = 16 * d['logits'].mean(-1)[o], d['true'][o]
            out[split] = dict(metrics(y, z - tau * LOGP), **{f'thr_{k}': v for k, v in threshold_free(z, y).items()})
    return out


def fmt(x, k):
    return f'{x[k]:.1f}' if x and k in x else '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tgz', nargs='*', default=[])
    ap.add_argument('--no-pav', action='store_true')
    a = ap.parse_args()
    for t in a.tgz:
        merge(t)

    names = sorted(p.name for p in LOGITS.iterdir()
                   if re.match(r'(T_|S_|ALL_)', p.name) and (p / 'val').is_dir() and (p / 'test').is_dir())
    names = ['B_strat_s0'] + names
    R = {}
    for n in names:
        rows = evaluate_run(LOGITS / n)
        rows['_probe'] = probe(LOGITS / n, rows)
        R[n] = rows

    L = ['# exp_2 phase 3: order-aware temporal models', '',
         'Written by `code/temporal_report.py`. Test = official 749 children; checkpoint and tau chosen on val '
         '(120/15/15). T_* runs are 4k-iteration fine-tunes of B_strat_s0@16k on ordered 30-frame clips, '
         'S_* runs are trained from scratch like B_strat_s0 (16k) on ordered clips. One row per run.', '',
         '## 1. Every run (val-tau rule)', '',
         '| run | iter | tau | acc | Sen | Spe | rec_neu | rec_pos | bal | AUC_sn | AUC_nn |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in R.items():
        m = r['val-tau']
        L.append(f"| {n} | {m['iter'] // 1000}k | {m['tau']:g} | {m['acc']:.1f} | {m['Sen']:.1f} | {m['Spe']:.1f} | "
                 f"{m['rec_neu']:.1f} | {m['rec_pos']:.1f} | {m['bal']:.1f} | {m['AUC_sn']:.1f} | {m['AUC_nn']:.1f} |")

    L += ['', '## 2. Probes at the last checkpoint (val tau kept)', '',
          'shuffle = the test frames in a random order. An order-free model gives exactly the normal numbers; '
          'a drop means the model uses the order of the frames. mirror / deskew as in exp_2 (lean shortcut).', '',
          '| run | bal normal | bal shuffled | AUC_sn normal | AUC_sn shuffled | AUC_sn mirror | AUC_sn deskew |',
          '|---|---|---|---|---|---|---|']
    for n, r in R.items():
        p = r['_probe']
        L.append(f"| {n} | {fmt(p.get('test'), 'bal')} | {fmt(p.get('test_shuffle'), 'bal')} | "
                 f"{fmt(p.get('test'), 'thr_AUC_sn')} | {fmt(p.get('test_shuffle'), 'thr_AUC_sn')} | "
                 f"{fmt(p.get('test_mirror'), 'thr_AUC_sn')} | {fmt(p.get('test_deskew'), 'thr_AUC_sn')} |")

    # 3. seeds: temporal vs control, paired by seed
    fam = {}
    for n in R:
        mm = re.match(r'(T_\w+?)_on_B_strat_s(\d)$', n)
        if mm:
            fam.setdefault(mm.group(1), {})[int(mm.group(2))] = R[n]
    if 'T_hppmax' in fam:
        L += ['', '## 3. Temporal vs control over seeds (fine-tunes of the same B_strat_s0; val-tau bal)', '',
              '| model | seeds | bal per seed | mean +- std | minus control, per seed | seeds better | bootstrap vs control [95% CI] |',
              '|---|---|---|---|---|---|---|']
        ctrl = fam['T_hppmax']
        for v in ['T_hppmax', 'T_tconv', 'T_pose']:
            if v not in fam:
                continue
            seeds = sorted(fam[v])
            b = [fam[v][s]['val-tau']['bal'] for s in seeds]
            row = f"| {v} | {','.join(map(str, seeds))} | {', '.join(f'{x:.1f}' for x in b)} | {np.mean(b):.1f} +- {np.std(b):.1f} |"
            common = [s for s in seeds if s in ctrl]
            if v != 'T_hppmax' and common:
                d = [fam[v][s]['val-tau']['bal'] - ctrl[s]['val-tau']['bal'] for s in common]
                g, lo, hi = paired_bootstrap([ctrl[s] for s in common], [fam[v][s] for s in common], 'val-tau')
                row += f" {', '.join(f'{x:+.1f}' for x in d)} | {sum(x > 0 for x in d)}/{len(d)} | {g:+.1f} [{lo:+.1f}, {hi:+.1f}] |"
            else:
                row += ' - | - | - |'
            L.append(row)

    # 4. from scratch vs B_strat_s0
    S = [n for n in R if n.startswith(('S_', 'ALL_'))]
    if S:
        L += ['', '## 4. vs B_strat_s0 (S_*: same sampler and schedule from scratch, but ordered clips + temporal conv; T_*_s0: fine-tunes of it)', '']
        for n in S + [n for n in R if n.startswith('T_') and n.endswith('_s0')]:
            g, lo, hi = paired_bootstrap([R['B_strat_s0']], [R[n]], 'val-tau')
            L.append(f'- {n}: {g:+.1f} [{lo:+.1f}, {hi:+.1f}] balanced accuracy vs B_strat_s0')
    L.append('\nReference: B_strat_s0 val-tau 89.7 bal; TG-MILNet (paper, full train set, 1 run) acc 89.9 / Sen 99.5 / '
             'Spe 86.4 / neutral 70.6.')
    out = EXP / 'analysis' / 'temporal.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf8')
    print('\n'.join(L))

    if not a.no_pav:   # PAV late fusion on the from-scratch runs (needs test_mirror at 16k)
        for n in S:
            it = R[n]['fixed']['iter']
            if (LOGITS / n / 'test_mirror' / f'iter_{it:05d}.npz').exists() and (LOGITS / n / 'val' / f'iter_{it:05d}.npz').exists():
                subprocess.run([sys.executable, str(HERE / 'pav_fusion.py'), '--run', n, '--iter', str(it)],
                               check=False, stdout=subprocess.DEVNULL)
                print(f'PAV fusion written: analysis/pav_fusion_{n}.md')


if __name__ == '__main__':
    main()
