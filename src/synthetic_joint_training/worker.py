"""Isolated GPU workers for single jobs or model-scoped baseline/joint phases."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .io import read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('train', 'eval', 'train-base', 'eval-base', 'train-joint', 'eval-joint'))
    p.add_argument('request', type=Path)
    request = p.parse_args()
    payload = read_json(request.request)
    if request.action in ('train-base', 'eval-base', 'train-joint', 'eval-joint'):
        from .base import train_bases, evaluate_bases
        (train_bases if request.action in ('train-base', 'train-joint') else evaluate_bases)(payload)
        return
    job, args = payload['job'], SimpleNamespace(**payload['args'])
    from .training_data import bundle_for_job, select_training_rows
    bundle = bundle_for_job(args.data_root, job)
    evaluation = bundle[3]
    if request.action == 'train':
        from .train import train
        rows, reference = select_training_rows(job, bundle)
        train(job, rows, payload['output'], resume=args.resume, reference=reference)
    else:
        from .evaluate import evaluate
        evaluate(job, evaluation, Path(payload['output']) / 'evaluations', payload['adapter'], args)


if __name__ == '__main__':
    main()
