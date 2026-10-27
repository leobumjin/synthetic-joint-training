"""Deterministic benchmark-only and joint-SFT plans; no GPU imports."""
from .data import RATIOS, count, synthetic_count
from .io import digest

# Keep the benchmark-only identity compatible with existing B100 artifacts.
PROTOCOL = 'sequential-direct-json-v2'
JOINT_PROTOCOL = 'joint-direct-json-v1'
BASE_FRACTIONS = (100,)


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


def reidentify(job):
    job['id'] = digest({k: v for k, v in job.items() if k not in ('id', 'n_examples', 'mode')})
    return job


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
                    baseline_recipe = recipe(args, rank)
                    base = make_job(model, args.revision, name, index['benchmark_id'], seed, 100, baseline_recipe)
                    base['n_examples'] = n
                    bases[base['id']] = base
                    if args.command == 'base':
                        jobs.append(base)
                        continue
                    pairs = []
                    if args.mode in ('all', 'replacement'):
                        pairs += [(100 - s, s) for s in RATIOS]
                    if args.mode in ('all', 'addition'):
                        pairs += [(100, s) for s in RATIOS]
                    for fraction, percent in pairs:
                        joint_recipe = {**baseline_recipe,
                            'epochs': args.synthetic_epochs if args.synthetic_epochs is not None else args.epochs,
                            'learning_rate': args.synthetic_learning_rate or args.learning_rate}
                        job = make_job(model, args.revision, name, index['benchmark_id'], seed, fraction,
                                       joint_recipe, index['synthetic_id'], percent)
                        n_benchmark = count(n, fraction)
                        n_synthetic = synthetic_count(n, fraction, percent)
                        job.update(protocol=JOINT_PROTOCOL, training_mode='joint',
                            initialization='original_model', optimizer_policy='fresh_per_run',
                            reference_id=base['id'], base_recipe=baseline_recipe,
                            n_benchmark_examples=n_benchmark, n_synthetic_examples=n_synthetic,
                            n_examples=n_benchmark + n_synthetic,
                            mode='replacement' if fraction < 100 else 'addition')
                        jobs.append(reidentify(job))
    return jobs, bases
