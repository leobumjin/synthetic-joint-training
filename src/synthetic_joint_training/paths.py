"""Readable artifact paths; run identities remain independent of storage layout."""
from pathlib import Path
import re


def component(value):
    return re.sub(r'[^A-Za-z0-9_.+-]', '_', str(value)) or 'unnamed'


def run_folder(root, job, migrate=False):
    root = Path(root)
    cfg = job['recipe']
    if job.get('kind') == 'unadapted':
        setting = 'original-model'
    else:
        setting = f"B{job['benchmark_percent']}"
        if job.get('synthetic_id'):
            setting += f"+S{job['synthetic_percent']}_{component(job.get('synthetic_name') or 'synthetic')}"
        if job.get('control_kind'):
            setting += '_' + component(job['control_kind'])
        setting += (f"_r{cfg['lora_rank']}_a{cfg['lora_alpha']}_seed{job['seed']}"
                    f"_ep{cfg['epochs']:g}_lr{cfg['learning_rate']:g}")
    folder = (root / 'runs' / component(job['benchmark']) / component(job['model']) /
              f"{setting}__{job['id'][:12]}")
    legacy = root / 'runs' / job['id']
    if migrate and legacy.is_dir() and not folder.exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(folder)
    return folder


def artifact_root(folder):
    """Find the output root for both legacy and readable artifact layouts."""
    folder = Path(folder).resolve()
    for parent in folder.parents:
        if parent.name == 'runs':
            return parent.parent
    raise ValueError(f'Artifact is outside a runs directory: {folder}')
