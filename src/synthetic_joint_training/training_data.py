"""Assemble frozen benchmark subsets and synthetic rows for joint SFT."""
import random

from .data import count, load_bundle, synthetic_count


def bundle_for_job(root, job):
    bundle = load_bundle(root, job['benchmark'], bool(job['synthetic_id']), job.get('synthetic_name'))
    index, manifest = bundle[:2]
    if manifest['id'] != job['benchmark_id']:
        raise ValueError('Dataset index changed after planning')
    if job['synthetic_id'] and index['synthetic_id'] != job['synthetic_id']:
        raise ValueError('Synthetic dataset index changed after planning')
    return bundle


def select_training_rows(job, bundle):
    _, manifest, benchmark, _, synthetic = bundle
    n = manifest['n_train']
    benchmark_rows = benchmark[:count(n, job['benchmark_percent'])]
    if not job['synthetic_id']:
        return benchmark_rows, None
    if job.get('training_mode') != 'joint' or job.get('parent_id'):
        raise ValueError('Synthetic training must be joint SFT initialized from the original model')
    amount = synthetic_count(n, job['benchmark_percent'], job['synthetic_percent'])
    if synthetic is None or len(synthetic) < amount:
        raise ValueError('Synthetic pool is too small for the joint condition')
    # Namespacing IDs prevents benchmark and synthetic row IDs from colliding.
    mixed = [{**row, 'id': 'benchmark:' + row['id'], 'training_source': 'benchmark'} for row in benchmark_rows]
    mixed += [{**row, 'id': 'synthetic:' + row['id'], 'training_source': 'synthetic'} for row in synthetic[:amount]]
    random.Random(job['seed']).shuffle(mixed)
    if job.get('control_kind') == 'benchmark-repeat':
        # Match the full joint run's token budget, starting from the original
        # model and training only on repeated benchmark rows.
        return benchmark, mixed
    return mixed, None
