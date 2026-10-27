"""One isolated GPU process per training or evaluation job."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .data import load_bundle, count, synthetic_count
from .io import read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('train', 'eval'))
    p.add_argument('request', type=Path)
    request = p.parse_args()
    payload = read_json(request.request)
    job, args = payload['job'], SimpleNamespace(**payload['args'])
    index, manifest, benchmark, evaluation, synthetic = load_bundle(args.data_root, job['benchmark'], bool(job['synthetic_id']), job.get('synthetic_name'))
    if manifest['id'] != job['benchmark_id']:
        raise ValueError('Dataset index changed after planning')
    if job['synthetic_id'] and index['synthetic_id'] != job['synthetic_id']:
        raise ValueError('Synthetic dataset index changed after planning')
    if request.action == 'train':
        from .train import train
        n = (synthetic_count(manifest['n_train'], job['benchmark_percent'], job['synthetic_percent'])
             if job['synthetic_id'] else count(manifest['n_train'], job['benchmark_percent']))
        rows = (synthetic if job['synthetic_id'] else benchmark)[:n]
        reference = rows if job.get('control_kind') == 'benchmark-repeat' else None
        if reference is not None:
            rows = benchmark
        train(job, rows, payload['output'],
              Path(payload['parent']) / 'adapter' if payload.get('parent') else None, args.resume, reference=reference)
    else:
        from .evaluate import evaluate
        evaluate(job, evaluation, Path(payload['output']) / 'evaluations', payload['adapter'], args)


if __name__ == '__main__':
    main()
