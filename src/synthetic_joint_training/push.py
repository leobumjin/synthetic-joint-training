"""Upload completed local LoRA artifacts independently of training and evaluation."""
import argparse
from pathlib import Path

from .io import lock, read_json, write_json
from .registry import register, repository_id, checkpoint_path, upload_artifact, verify_local, DEFAULT_BASE_REPO, DEFAULT_SYNTHETIC_REPO


def discover(output_root, registry):
    folders = {p.parent.resolve() for p in Path(output_root).glob('runs/*/training.json')}
    if Path(registry).exists():
        for entry in read_json(registry)['adapters'].values():
            if entry.get('local_path'):
                folder = (Path(registry).parent / entry['local_path']).resolve()
                if folder.exists():
                    folders.add(folder)
    result = {}
    for folder in sorted(folders):
        state = read_json(folder / 'training.json')
        if state.get('status') != 'complete':
            continue
        job = state['job']
        if job['id'] in result and result[job['id']][1]['adapter_sha256'] != state['adapter_sha256']:
            raise ValueError(f"Conflicting local weights for {job['id']}")
        result[job['id']] = (folder, state)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path('outputs'))
    p.add_argument('--registry', type=Path, default=Path('configs/base_lora_registry.json'))
    p.add_argument('--hf-namespace', required=True, help='HF username or organization')
    p.add_argument('--hf-repo', default=DEFAULT_BASE_REPO, help='Shared base LoRA repository name')
    p.add_argument('--hf-synthetic-repo', default=DEFAULT_SYNTHETIC_REPO, help='Shared synthetic LoRA repository name')
    p.add_argument('--kind', choices=('base', 'synthetic', 'all'), default='base')
    p.add_argument('--models', nargs='+')
    p.add_argument('--benchmarks', nargs='+')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--job-ids', nargs='+', help='Exact run IDs to upload')
    p.add_argument('--private', action='store_true', help='Create new model repositories as private')
    p.add_argument('--dry-run', action='store_true', help='Verify local files and print destinations; no HF calls')
    args = p.parse_args(argv)
    # Validate the namespace even if no completed checkpoints exist.
    repository_id({'model': 'model', 'benchmark': 'task', 'benchmark_percent': 100,
                   'seed': 0, 'id': 'a'*64}, args.hf_namespace, args.hf_repo)
    repository_id({'synthetic_id': 'synthetic'}, args.hf_namespace, args.hf_synthetic_repo)
    selected = []
    for key, (folder, state) in discover(args.output_root, args.registry).items():
        job = state['job']
        kind = 'synthetic' if job.get('synthetic_id') else 'base'
        if args.kind != 'all' and args.kind != kind:
            continue
        if args.models and job['model'] not in args.models:
            continue
        if args.benchmarks and job['benchmark'] not in args.benchmarks:
            continue
        if args.seeds is not None and job['seed'] not in args.seeds:
            continue
        if args.job_ids and key not in args.job_ids:
            continue
        verify_local(folder, job)
        selected.append((folder, state, kind))
    if args.job_ids:
        missing = set(args.job_ids) - {state['job']['id'] for _, state, _ in selected}
        if missing:
            p.error('Requested completed local jobs not found under selected filters: ' + ', '.join(sorted(missing)))
    if not selected:
        print('No matching completed local LoRA checkpoints. Nothing uploaded.')
        return
    for folder, state, kind in selected:
        job = state['job']
        print(f"{kind}: {folder}\n  -> https://huggingface.co/{repository_id(job, args.hf_namespace, args.hf_repo if kind == 'base' else args.hf_synthetic_repo)}/tree/main/{checkpoint_path(job)}\n"
              '  files: adapter/*, training.json', flush=True)
    if args.dry_run:
        print(f'Dry run: {len(selected)} checkpoint(s); no HF requests made.')
        return
    for folder, state, kind in selected:
        job = state['job']
        with lock(args.output_root / '.locks' / f"{job['id']}.lock"):
            if verify_local(folder, job) != state:
                raise ValueError('Training metadata changed after upload planning')
            if kind == 'base':
                register(args.registry, job, folder)
            uploaded = upload_artifact(folder, job, args.hf_namespace, args.private,
                                       args.hf_repo if kind == 'base' else args.hf_synthetic_repo)
            # Persist each success immediately so interrupted batches retain commit IDs.
            catalog = args.output_root / 'hf_uploads.json'
            with lock(catalog.with_suffix('.lock')):
                data = read_json(catalog) if catalog.exists() else {'schema': 1, 'adapters': {}}
                data['adapters'].setdefault(job['id'], {})[uploaded['repo_id']] = {
                    **uploaded, 'job': job, 'adapter_sha256': state['adapter_sha256']}
                write_json(catalog, data)
            if kind == 'base':
                with lock(args.registry.with_suffix('.lock')):
                    data = read_json(args.registry)
                    data['adapters'][job['id']].update(uploaded)
                    write_json(args.registry, data)
    print(f'Uploaded {len(selected)} checkpoint(s). Commit IDs: {args.output_root / "hf_uploads.json"}')


if __name__ == '__main__':
    main()
