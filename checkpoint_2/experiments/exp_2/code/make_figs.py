"""PPT figures for exp_2 from the merged logits (rerun when new runs arrive).

python checkpoint_2/experiments/exp_2/code/make_figs.py [--root checkpoint_2/experiments/exp_2/runs/logits] -> checkpoint_2/experiments/exp_2/analysis/figs/*.png + bootstrap.md
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent            # exp_2/code
EXP = HERE.parent                                   # checkpoint_2/experiments/exp_2
sys.path.insert(0, str(HERE))
sys.argv, _argv = sys.argv[:1], sys.argv
import calibrate as cal  # noqa: E402
sys.argv = _argv

# reference palette (dataviz skill, light mode), fixed order per entity
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
NAMES = {'A_base': 'A  ScoNet-MT', 'B_strat': 'B  stratified', 'C_up': 'C  upsample', 'D_la': 'D  LA loss',
         'E_bam_on_D_la': 'E  D + BAM', 'F_ord_on_D_la': 'F  D + ordinal', 'G_deskew_on_D_la': 'G  D + deskew',
         'L_flip_on_D_la': 'L  D + flip', 'F_ord_on_B_strat': 'F*  B + ordinal'}
ORDER = list(NAMES)

plt.rcParams.update({'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'legend.frameon': False})


def grouped_bars(ax, variants, series, colors, labels, ylim, fmt='{:.1f}'):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(variants))
    for j, (vals, col, lab) in enumerate(zip(series, colors, labels)):
        xs = x - 0.4 + w * (j + 0.5)
        ax.bar(xs, [v - ylim[0] for v in vals], w - 0.03, bottom=ylim[0], color=col, label=lab, zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + (ylim[1] - ylim[0]) * 0.01, fmt.format(v), ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_xticks(x, [NAMES[v] for v in variants], rotation=0, fontsize=10)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(EXP / 'runs' / 'logits'))
    a = ap.parse_args()
    out = EXP / 'analysis' / 'figs'
    out.mkdir(parents=True, exist_ok=True)

    by = {}
    for r in sorted(p for p in Path(a.root).iterdir() if (p / 'val').is_dir()):
        rows = cal.evaluate_run(r)
        rows['_beyond'] = cal.beyond_accuracy(r, rows)
        by.setdefault(re.sub(r'_s\d+$', '', r.name), []).append(rows)
    vs = [v for v in ORDER if v in by]
    mean = lambda v, rule, k: np.mean([r[rule][k] for r in by[v]])

    # 1. threshold: fixed (tau 0) vs val-chosen tau
    fig, ax = plt.subplots(figsize=(11, 4.8))
    grouped_bars(ax, vs, [[mean(v, 'fixed', 'bal') for v in vs], [mean(v, 'val-tau', 'bal') for v in vs]],
                 [C[1], C[0]], ['default threshold (argmax)', 'threshold chosen on val'], (60, 95))
    ax.set_ylabel('balanced accuracy on test (%)')
    ax.set_title('One val-chosen threshold closes most of the gap', loc='left', color=INK)
    ax.legend(loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig(out / 'fig1_threshold.png', dpi=200); plt.close(fig)

    # 2. neutral recall (the hard class) vs TG-MILNet
    fig, ax = plt.subplots(figsize=(11, 4.8))
    grouped_bars(ax, vs, [[mean(v, 'fixed', 'rec_neu') for v in vs], [mean(v, 'val-tau', 'rec_neu') for v in vs]],
                 [C[1], C[0]], ['default threshold', 'threshold chosen on val'], (0, 100))
    ax.axhline(70.6, color=INK2, lw=1.5, ls='--', zorder=3)
    ax.text(len(vs) - 0.5, 72, 'TG-MILNet 70.6', ha='right', color=INK2, fontsize=10)
    ax.set_ylabel('neutral recall on test (%)')
    ax.set_title('Neutral (mild) children: recall per variant', loc='left', color=INK)
    ax.legend(loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig(out / 'fig2_neutral.png', dpi=200); plt.close(fig)

    # 3. lean shortcut: AUC on normal / mirrored / deskewed test
    probe_vs = [v for v in vs if any('test_mirror' in r['_beyond'] for r in by[v])]
    auc = lambda v, sp: np.mean([r['_beyond'][sp]['AUC_sn'] for r in by[v] if sp in r['_beyond']])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    grouped_bars(ax, probe_vs, [[auc(v, sp) for v in probe_vs] for sp in ('test', 'test_mirror', 'test_deskew')],
                 [C[0], C[1], C[2]], ['normal', 'mirrored left/right', 'lean removed (deskew)'], (0, 105), '{:.0f}')
    ax.axhline(50, color=INK2, lw=1.5, ls='--', zorder=3)
    ax.text(len(probe_vs) - 0.5, 52, 'chance', ha='right', color=INK2, fontsize=10)
    ax.set_ylabel('AUC scoliosis vs negative (%)')
    ax.set_title('Mirroring the input flips the models: they read the body lean', loc='left', color=INK)
    ax.legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.06))
    fig.tight_layout(); fig.savefig(out / 'fig3_mirror.png', dpi=200); plt.close(fig)

    # 4. learning curves (test bal, tau chosen on val per checkpoint)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, v in enumerate(vs):
        cs = [r['_beyond']['curve'] for r in by[v]]
        n = min(len(c) for c in cs)
        it = [cs[0][k]['iter'] / 1000 for k in range(n)]
        y = [np.mean([c[k]['test_bal'] for c in cs]) for k in range(n)]
        col = C[ORDER.index(v) % len(C)]
        ax.plot(it, y, color=col, lw=2, marker='o', ms=6, label=NAMES[v])
    for m in (10, 14):
        ax.axvline(m, color=GRID, lw=1.5, zorder=0)
        ax.text(m, 31, f' lr drop {m}k', color=INK2, fontsize=9)
    ax.set_xlabel('iteration (k)'); ax.set_ylabel('balanced accuracy on test (%)')
    ax.set_ylim(30, 95)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_title('All variants become good only after the lr drops; checkpoints jump a lot', loc='left', color=INK)
    ax.legend(loc='upper left', ncol=3, fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'fig4_curves.png', dpi=200); plt.close(fig)

    # 5. paired bootstrap vs A (val-tau and val-fuse)
    ref = 'A_base'
    lines = ['| variant | val-tau gain [95% CI] | val-fuse gain [95% CI] |', '|---|---|---|']
    res = {}
    for v in vs:
        if v == ref or len(by[v]) != len(by[ref]):
            continue
        res[v] = [cal.paired_bootstrap(by[ref], by[v], rule) for rule in ('val-tau', 'val-fuse')]
        lines.append(f'| {v} | ' + ' | '.join(f'{g:+.1f} [{lo:+.1f}, {hi:+.1f}]' for g, lo, hi in res[v]) + ' |')
    (EXP / 'analysis' / 'bootstrap.md').write_text('\n'.join(lines), encoding='utf8')
    print('\n'.join(lines))
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(res) + 1.6))
    for i, (v, ((g, lo, hi), _)) in enumerate(res.items()):
        col = C[0] if lo > 0 else (C[7] if hi < 0 else INK2)
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(g, i, 'o', color=col, ms=9)
        right = hi >= 0
        ax.text(hi + 0.3 if right else lo - 0.3, i, f'{g:+.1f}  [{lo:+.1f}, {hi:+.1f}]', va='center',
                ha='left' if right else 'right', fontsize=10, color=INK)
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_yticks(range(len(res)), [NAMES[v] for v in res])
    ax.invert_yaxis()
    ax.set_xlabel(f'balanced-accuracy gain vs {NAMES[ref].split()[-1]} (points, threshold chosen on val)')
    ax.set_xlim(min(lo for (_, lo, _), _ in res.values()) - 7,max(hi for (_, _, hi), _ in res.values()) + 6)
    ax.set_title('Paired bootstrap, 95% CI (1 seed each)', loc='left', color=INK)
    fig.tight_layout(); fig.savefig(out / 'fig5_bootstrap.png', dpi=200); plt.close(fig)
    print('figures in', out)


if __name__ == '__main__':
    main()
