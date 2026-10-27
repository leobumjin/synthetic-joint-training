"""Model-scoped baseline and joint training/evaluation phases with nested progress bars."""
from pathlib import Path
from types import SimpleNamespace

from .training_data import bundle_for_job, select_training_rows
from .io import lock, write_json
from .registry import register, resolve, verify_local


def description(job):
    if job.get('kind') == 'unadapted':
        return f"{job['model'].split('/')[-1]} | {job['benchmark']} | original model"
    cfg = job['recipe']
    return (f"{job['model'].split('/')[-1]} | {job['benchmark']} | B{job['benchmark_percent']} | rank={cfg['lora_rank']} "
            f"alpha={cfg['lora_alpha']} | seed={job['seed']}"
            + (f" | +S{job['synthetic_percent']} joint" if job['synthetic_id'] else ''))


def train_bases(payload):
    from tqdm.auto import tqdm
    from .train import TrainingSession, cleanup_checkpoints, completed, train
    args = SimpleNamespace(**payload['args'])
    jobs = payload['jobs']
    is_joint = bool(jobs[0]['synthetic_id'])
    root = Path(args.output_root if is_joint else args.base_output_root)
    session, bundles = TrainingSession(), {}
    with tqdm(jobs, desc='Joint LoRA training' if is_joint else 'Base LoRA training', position=0, unit='LoRA') as progress:
        for job in progress:
            progress.set_description(('Joint training | ' if is_joint else 'Base training | ') + description(job))
            folder = root / 'runs' / job['id']
            with lock(root / '.locks' / f"{job['id']}.lock"):
                write_json(folder / 'job.json', job)
                cached = completed(job, folder)
                if not cached:
                    key = (job['benchmark'], job.get('synthetic_name'), job['synthetic_id'])
                    if key not in bundles:
                        bundles[key] = bundle_for_job(args.data_root, job)
                    selected, reference = select_training_rows(job, bundles[key])
                    train(job, selected, folder, resume=args.resume, reference=reference, session=session)
                if cached:
                    cleanup_checkpoints(job, folder)
                if not is_joint:
                    register(args.registry, job, folder, reconcile=True)
                progress.set_postfix(status='cached' if cached else 'trained')


def evaluate_bases(payload):
    from tqdm.auto import tqdm
    from .evaluate import EvaluationSession, evaluate
    args = SimpleNamespace(**payload['args'])
    jobs = payload['jobs']
    is_joint = bool(jobs[0]['synthetic_id'])
    session = EvaluationSession(max_rank=max(j['recipe']['lora_rank'] for j in jobs))
    bundles = {}
    def run(job):
        key = (job['benchmark'], job.get('synthetic_name'), job['synthetic_id'])
        if key not in bundles:
            bundles[key] = bundle_for_job(args.data_root, job)
        _, manifest, _, evaluation, _ = bundles[key]
        if manifest['id'] != job['benchmark_id']:
            raise ValueError('Dataset index changed after planning')
        root = Path(args.output_root if job['synthetic_id'] else args.base_output_root)
        folder = root / 'runs' / job['id']
        with lock(root / '.locks' / f"{job['id']}.lock"):
            if job.get('kind') == 'unadapted':
                adapter = None
            elif job['synthetic_id']:
                verify_local(folder, job)
                adapter = folder / 'adapter'
            else:
                adapter = resolve(args.registry, job) / 'adapter'
            return evaluate(job, evaluation, folder / 'evaluations', adapter, args, session=session)
    # Original references share the inference engine but are not counted as LoRAs.
    for job in payload.get('originals', []):
        tqdm.write('Reference evaluation | ' + description(job))
        run(job)
    for job in payload.get('references', []):
        tqdm.write('B100 reference evaluation | ' + description(job))
        run(job)
    with tqdm(jobs, desc='Joint LoRA evaluation' if is_joint else 'Base LoRA evaluation', position=0, unit='LoRA') as progress:
        for job in progress:
            progress.set_description(('Joint evaluation | ' if is_joint else 'Base evaluation | ') + description(job))
            report = run(job)
            progress.set_postfix({style: f"{metrics['accuracy']:.2%}" for style, metrics in report['metrics'].items()})
