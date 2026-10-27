"""Deterministic plans; no model libraries or GPUs required."""
from .data import FRACTIONS, RATIOS, count, synthetic_count
from .io import digest

PROTOCOL = 'sequential-direct-json-v1'


def ranks(args):
    return [args.lora_rank] if args.lora_rank is not None else args.lora_ranks


def recipe(args, rank=None):
    rank = ranks(args)[0] if rank is None else rank
    result = {key: getattr(args, key) for key in ('epochs', 'learning_rate', 'batch_size',
        'gradient_accumulation', 'max_length', 'lora_dropout',
        'target_modules', 'dtype', 'warmup_ratio', 'weight_decay')}
    return {**result, 'lora_rank': rank,
            'lora_alpha': args.lora_alpha if args.lora_alpha is not None else 2 * rank}


def make_job(model, revision, benchmark, data_id, seed, fraction, training, synthetic_id=None,
             synthetic_percent=0):
    identity = {'protocol': PROTOCOL, 'model': model, 'revision': revision, 'benchmark': benchmark,
                'benchmark_id': data_id, 'seed': seed, 'benchmark_percent': fraction,
                'synthetic_percent': synthetic_percent, 'synthetic_id': synthetic_id,
                'recipe': training, 'optimizer_policy': 'fresh_per_stage',
                'loss_policy': 'final_assistant_only', 'packing': False,
                'truncation': 'reject_overlength'}
    return {'id': digest(identity), **identity}


def build_plan(args, bundles):
    jobs, bases = [], {}
    for rank in ranks(args):
        for model in args.models:
            for name, bundle in bundles.items():
                index, manifest = bundle[:2]
                n = manifest['n_train']
                if count(n, min(RATIOS)) < 1:
                    raise ValueError('At least 10 benchmark rows are required')
                for seed in args.seeds:
                    for fraction in FRACTIONS:
                        base = make_job(model, args.revision, name, index['benchmark_id'], seed, fraction, recipe(args, rank))
                        base['n_examples'] = count(n, fraction)
                        bases[base['id']] = base
                        if args.command == 'base':
                            jobs.append(base)
                    if args.command == 'synthetic':
                        pairs = []
                        if args.mode in ('all', 'replacement'):
                            pairs += [(100 - s, s) for s in RATIOS]
                        if args.mode in ('all', 'addition'):
                            pairs += [(100, s) for s in RATIOS]
                        for fraction, percent in pairs:
                            base = make_job(model, args.revision, name, index['benchmark_id'], seed, fraction, recipe(args, rank))
                            job = make_job(model, args.revision, name, index['benchmark_id'], seed, fraction,
                                           recipe(args, rank), index['synthetic_id'], percent)
                            job.update(parent_id=base['id'], n_examples=synthetic_count(n, fraction, percent),
                                       mode='replacement' if fraction < 100 else 'addition')
                            jobs.append(job)
    return jobs, bases
