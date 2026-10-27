"""CLI for benchmark-only references and independent joint-SFT runs."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

from .data import load_bundle, synthetic_name
from .io import read_json, write_json
from .plan import build_plan, digest, ranks, reidentify
from .registry import publish, registry_entry, DEFAULT_BASE_REPO


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('base', 'synthetic'))
    p.add_argument('--models', nargs='+', default=['Qwen/Qwen3-4B-Instruct-2507'])
    p.add_argument('--revision', default='main', help='Resolved once to immutable model commit before execution')
    p.add_argument('--benchmarks', nargs='+', default=['gsm8k'])
    p.add_argument('--synthetic', nargs='*', help='Prepared dataset names or name=file specs; default: all prepared datasets')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--control', choices=('none', 'benchmark-repeat'), default='none',
                   help='Addition arms only: repeat benchmark to nearest whole-example full joint input-token budget')
    p.add_argument('--mode', choices=('replacement', 'addition', 'all'), default='all')
    p.add_argument('--stage', choices=('train', 'eval', 'all'), default='all')
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output-root', type=Path, default=Path('outputs'),
                   help='Joint SFT output root (also searched for legacy mixed runs)')
    p.add_argument('--base-output-root', type=Path, default=Path('outputs_base_lora'),
                   help='Base LoRA and reference evaluation output root')
    p.add_argument('--registry', type=Path, default=Path('configs/base_lora_registry.json'))
    p.add_argument('--epochs', type=float, default=5.0, help='Benchmark checkpoint recipe; must match registry')
    p.add_argument('--joint-epochs', '--synthetic-epochs', dest='synthetic_epochs', type=float,
                   help='Joint SFT epochs; default: same as benchmark-only --epochs')
    p.add_argument('--joint-learning-rate', '--synthetic-learning-rate', dest='synthetic_learning_rate',
                   type=float, default=None, help='Joint SFT learning rate; default: same as benchmark')
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--gradient-accumulation', type=int, default=2)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--lora-ranks', type=int, nargs='+', default=[16, 32])
    p.add_argument('--lora-rank', type=int, help='Select one rank; overrides --lora-ranks')
    p.add_argument('--lora-alpha', type=int, help='Default: 2 * each rank (16/32 and 32/64)')
    p.add_argument('--lora-dropout', type=float, default=0.0)
    p.add_argument('--target-modules', default='all-linear')
    p.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    p.add_argument('--warmup-ratio', type=float, default=0.0)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--eval-split', choices=('test', 'validation'), default='test')
    p.add_argument('--prompt-styles', nargs='+', choices=('Direct', 'CoT'), default=['Direct', 'CoT'])
    p.add_argument('--direct-max-tokens', type=int, default=256)
    p.add_argument('--cot-max-tokens', type=int, default=2048)
    p.add_argument('--eval-batch-size', type=int, default=64, help='Prompts per inference batch/progress update')
    p.add_argument('--eval-seed', type=int, default=42)
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--max-model-len', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--push-to-hub', action='store_true')
    p.add_argument('--hf-namespace')
    p.add_argument('--hf-repo', default=DEFAULT_BASE_REPO)
    p.add_argument('--base-lora-repo', help='Download registry.json only from this HF repo (synthetic command)')
    p.add_argument('--base-lora-revision', default='main', help='Remote registry revision; adapter revisions remain pinned')
    p.add_argument('--train-python', default=sys.executable)
    p.add_argument('--inference-python', default=sys.executable)
    return p


def validate(p, args):
    positive = ('epochs', 'learning_rate', 'batch_size', 'gradient_accumulation',
                'max_length', 'eval_batch_size', 'direct_max_tokens', 'cot_max_tokens', 'tensor_parallel_size')
    if any(getattr(args, k) <= 0 for k in positive) or any(s < 0 for s in args.seeds):
        p.error('Training sizes/rates must be positive and seeds nonnegative')
    if any(rank <= 0 for rank in ranks(args)) or len(set(ranks(args))) != len(ranks(args)):
        p.error('LoRA ranks must be positive and unique')
    if args.lora_alpha is not None and args.lora_alpha <= 0:
        p.error('LoRA alpha must be positive')
    if args.synthetic_epochs is not None and args.synthetic_epochs <= 0:
        p.error('Joint epochs must be positive')
    if args.synthetic_learning_rate is not None and args.synthetic_learning_rate <= 0:
        p.error('Synthetic learning rate must be positive')
    if not 0 <= args.lora_dropout < 1 or not 0 <= args.warmup_ratio < 1 or args.weight_decay < 0:
        p.error('Invalid dropout/warmup/weight decay')
    if not 0 < args.gpu_memory_utilization <= 1 or (args.max_model_len is not None and args.max_model_len < 1):
        p.error('Invalid inference memory/context limit')
    if args.push_to_hub and (args.command != 'base' or not args.hf_namespace):
        p.error('--push-to-hub requires base command and --hf-namespace')
    if args.base_lora_repo and args.command != 'synthetic':
        p.error('--base-lora-repo is for synthetic continuation only')
    if args.command == 'synthetic' and args.control != 'none' and args.mode == 'replacement':
        p.error('benchmark-repeat control is available for addition/all modes')
    if len(set(args.prompt_styles)) != len(args.prompt_styles) or len(set(args.benchmarks)) != len(args.benchmarks):
        p.error('Duplicate prompt styles/benchmarks are not allowed')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate models/seeds are not allowed')


def spawn(action, payload, interpreter, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = folder / f'{action}-request.json'
    write_json(request, payload)
    subprocess.run([interpreter, '-m', 'synthetic_joint_training.worker', action, str(request)], check=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    validate(p, args)
    baseline_bundles = {name: load_bundle(args.data_root, name) for name in args.benchmarks}
    if args.command == 'base':
        selections = [None]
    elif args.synthetic is not None:
        selections = [synthetic_name(spec) for spec in args.synthetic]
        if not selections or len(set(selections)) != len(selections):
            p.error('Select at least one unique synthetic dataset')
    else:
        labels = sorted(set().union(*(b[0].get('synthetics', {}) for b in baseline_bundles.values())))
        selections = labels or [None]  # Backward-compatible unnamed single pool.
    # Validate every requested dataset before launching the first expensive job.
    prepared = [(selection, {name: load_bundle(args.data_root, name, args.command == 'synthetic', selection)
                             for name in args.benchmarks}) for selection in selections]
    for selection, bundles in prepared:
        current = copy.copy(args)
        current.synthetic_selection = selection
        execute(current, p, bundles)


def execute(args, p, bundles):
    active_root = args.base_output_root if args.command == 'base' else args.output_root
    if args.stage in ('eval', 'all'):
        for name, bundle in bundles.items():
            if not any(r['split'] == args.eval_split for r in bundle[3]):
                p.error(f'{name} has no {args.eval_split} split; choose a compatible evaluation split')
    if args.base_lora_repo and args.stage in ('eval', 'all') and not args.dry_run:
        from huggingface_hub import hf_hub_download
        args.registry = Path(hf_hub_download(repo_id=args.base_lora_repo, filename='registry.json',
                             revision=args.base_lora_revision, repo_type='model'))
    # Every actual execution records a fixed model revision; planning is network-free.
    requested_revision = args.revision
    models = args.models[:]
    for model in models:
        args.models = [model]
        if not args.dry_run and not re.fullmatch('[0-9a-f]{40}', args.revision):
            from huggingface_hub import HfApi
            args.revision = HfApi().model_info(model, revision=args.revision).sha
        jobs, bases = build_plan(args, bundles)
        if args.command == 'synthetic':
            for job in jobs:
                if args.synthetic_selection is not None:
                    job['synthetic_name'] = args.synthetic_selection
                    reidentify(job)
        if args.command == 'synthetic' and args.control == 'benchmark-repeat':
            controls = []
            for job in jobs:
                if job['mode'] == 'addition':
                    control = {**job, 'control_kind': 'benchmark-repeat', 'mode': 'addition-control',
                               'budget_policy': 'nearest_whole_example_input_tokens'}
                    reidentify(control)
                    controls.append(control)
            jobs += controls
        if args.dry_run:
            print(json.dumps({'model': model, 'model_revision': args.revision,
                'revision_note': 'Mutable revisions are resolved at execution time',
                'output_root': str(active_root), 'base_output_root': str(args.base_output_root),
                'command': args.command, 'synthetic': args.synthetic_selection, 'stage': args.stage, 'jobs': jobs}, indent=2))
            continue
        active_root.mkdir(parents=True, exist_ok=True)
        if args.command == 'synthetic' and args.stage in ('eval', 'all'):
            # B100 is an evaluation reference, never an initialization checkpoint.
            for reference in bases.values():
                registry_entry(args.registry, reference)
        if args.stage in ('train', 'all'):
            plan_identity = {'command': args.command, 'jobs': jobs}
            write_json(active_root / 'plans' / f'{digest(plan_identity)}.json', plan_identity)
        payload_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        phase_folder = active_root / 'phases' / digest({'jobs': [j['id'] for j in jobs]})
        phase_payload = {'jobs': jobs, 'args': payload_args}
        if args.stage in ('train', 'all'):
            spawn('train-base' if args.command == 'base' else 'train-joint',
                  phase_payload, args.train_python, phase_folder)
        if args.stage in ('eval', 'all'):
            for reference in bases.values():
                registry_entry(args.registry, reference)
            originals = []
            for name in bundles:
                exemplar = next(j for j in bases.values() if j['benchmark'] == name)
                original = {**exemplar, 'benchmark_percent': 0, 'synthetic_percent': 0,
                            'seed': None, 'kind': 'unadapted'}
                original['id'] = digest({k: v for k, v in original.items() if k != 'id'})
                originals.append(original)
            spawn('eval-base' if args.command == 'base' else 'eval-joint',
                  {**phase_payload, 'originals': originals,
                   'references': list(bases.values()) if args.command == 'synthetic' else []},
                  args.inference_python, phase_folder)
        if args.push_to_hub:
            for reference in bases.values():
                publish(args.registry, reference, args.hf_namespace, repo_name=args.hf_repo)
        # Restore requested revision so each model resolves its own commit.
        args.revision = requested_revision


if __name__ == '__main__':
    main()
