import argparse
import glob
import os
import re
import subprocess
import sys
import zipfile

REPRO = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(REPRO, 'datasets')
PKL_DIR = os.path.join(DATASETS, 'Scoliosis1K-sil-pkl')
N_SEQUENCES = 1493

VARIANTS = {
    'sconet_mt':      ('model_sconet', 'sconet_mt.yaml',      'ScoNet',         'ScoNet_MT'),
    'sconet_triplet': ('model_sconet', 'sconet_triplet.yaml', 'ScoNet_Triplet', 'ScoNet_Triplet'),
    'sconet':         ('model_sconet', 'sconet.yaml',         'ScoNet_CE',      'ScoNet_CE'),
    'cnn_mt':         ('model_cnn',    'cnn_mt.yaml',         'CNN_MT',         'CNN_MT'),
    'cnn_triplet':    ('model_cnn',    'cnn_triplet.yaml',    'CNN_Triplet',    'CNN_Triplet'),
    'cnn':            ('model_cnn',    'cnn.yaml',            'CNN_CE',         'CNN_CE'),
}


def sh(cmd, cwd=None, env=None):
    print('+', cmd if isinstance(cmd, str) else ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True, shell=isinstance(cmd, str))


def count_pkl():
    return len(glob.glob(os.path.join(PKL_DIR, '*', '*', '*', '*.pkl')))


def bundle(_):
    out = os.path.join(REPRO, 'reproduce_code.zip')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ['runpod.py', 'requirements.txt']:
            z.write(os.path.join(REPRO, f), f'reproduce/{f}')
        for f in glob.glob(os.path.join(DATASETS, '*.json')):
            z.write(f, f'reproduce/datasets/{os.path.basename(f)}')
        for folder in ['model_sconet', 'model_cnn']:
            for root, dirs, files in os.walk(os.path.join(REPRO, folder)):
                dirs[:] = [d for d in dirs if d not in ('output', '__pycache__')]
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.join('reproduce', os.path.relpath(p, REPRO)))
    print(f'Wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB). Upload it with datasets/Scoliosis1K-sil-pkl.zip.')


def setup(_):
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
    sh([sys.executable, '-m', 'pip', 'install', '-q', '-r', os.path.join(REPRO, 'requirements.txt')])
    if count_pkl() != N_SEQUENCES:
        zip_path = os.path.join(DATASETS, 'Scoliosis1K-sil-pkl.zip')
        if not os.path.isfile(zip_path):
            sys.exit(f'Upload Scoliosis1K-sil-pkl.zip to {DATASETS} first.')
        if subprocess.run('command -v unzip', shell=True).returncode != 0:
            sh('apt-get update -qq && apt-get install -y -qq unzip')
        sh(['unzip', '-q', '-o', zip_path, '-d', DATASETS])
    print(f'pkl sequences found: {count_pkl()} (expected {N_SEQUENCES})')
    sh([sys.executable, '-c', "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"])


def opengait(variant, phase, iteration=None):
    folder, cfg, _, _ = VARIANTS[variant]
    if count_pkl() != N_SEQUENCES:
        sys.exit('Dataset not ready. Run: python runpod.py setup')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    cmd = ['torchrun', '--nproc_per_node=1', 'opengait/main.py',
           '--cfgs', f'./configs/{cfg}', '--phase', phase, '--log_to_file']
    if iteration:
        cmd += ['--iter', str(iteration)]
    sh(cmd, cwd=os.path.join(REPRO, folder), env=env)


def run_dir(variant):
    folder, _, model, save = VARIANTS[variant]
    return os.path.join(REPRO, folder, 'output', 'Scoliosis1K', model, save)


def checkpoints(variant):
    found = glob.glob(os.path.join(run_dir(variant), 'checkpoints', '*.pt'))
    return sorted(int(re.search(r'-(\d+)\.pt$', p).group(1)) for p in found)


def train(args):
    opengait(args.variant, 'train')


def test(args):
    opengait(args.variant, 'test', args.iter)


def sweep(args):
    its = checkpoints(args.variant)
    if not its:
        sys.exit('No checkpoints yet. Train first.')
    for it in its:
        opengait(args.variant, 'test', it)
    print('Each test wrote a log file to', os.path.join(run_dir(args.variant), 'logs'))


def pack(args):
    src = run_dir(args.variant)
    if not os.path.isdir(src):
        sys.exit(f'Nothing to pack: {src} does not exist.')
    folder, cfg, model, save = VARIANTS[args.variant]
    out = os.path.join(REPRO, f'results_{save}.zip')
    base = os.path.join(REPRO, folder)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src):
            if not args.with_summary and os.path.basename(root) == 'summary':
                continue  # TensorBoard files are ~150 MB; the text logs hold the same numbers
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, base))  # keeps the output/... path
        z.write(os.path.join(base, 'configs', cfg), f'config_used/{cfg}')
    print(f'Wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB).')
    print(f'Download it, then extract into reproduce/result_{folder.split("_")[1]}/{save}/')


def run_all(args):
    train(args)
    sweep(args)
    pack(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('bundle').set_defaults(fn=bundle)
    sub.add_parser('setup').set_defaults(fn=setup)
    for name, fn in [('train', train), ('test', test), ('sweep', sweep), ('pack', pack), ('all', run_all)]:
        s = sub.add_parser(name)
        s.add_argument('variant', choices=VARIANTS)
        s.add_argument('--iter', type=int, default=None, help='checkpoint to test (default: restore_hint 20000)')
        s.add_argument('--with-summary', action='store_true', help='also zip the TensorBoard summary folder')
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
