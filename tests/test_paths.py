"""Readable storage retains immutable identities and discovers both layouts."""
from synthetic_joint_training.cli import parser
from synthetic_joint_training.io import write_json
from synthetic_joint_training.paths import artifact_root, run_folder
from synthetic_joint_training.plan import make_job, recipe
from synthetic_joint_training.push import discover
from synthetic_joint_training.train import adapter_signature


def fixture_job():
    args = parser().parse_args(['base'])
    return make_job('google/gemma-3-4b-it', 'a'*40, 'gsm8k', 'data', 0, 100, recipe(args, 16))


def test_readable_path_and_legacy_migration(tmp_path):
    job = fixture_job()
    folder = run_folder(tmp_path, job)
    assert folder.parent == tmp_path / 'runs/gsm8k/google_gemma-3-4b-it'
    assert folder.name.startswith('B100_r16_a32_seed0_ep5_lr2e-05__')
    legacy = tmp_path / 'runs' / job['id']
    write_json(legacy / 'job.json', job)
    assert run_folder(tmp_path, job, migrate=True) == folder
    assert (folder / 'job.json').exists() and not legacy.exists()
    assert artifact_root(folder) == tmp_path.resolve()
    assert artifact_root(legacy) == tmp_path.resolve()
    assert job['id'] == fixture_job()['id']


def test_upload_discovers_readable_runs(tmp_path):
    job = fixture_job()
    folder = run_folder(tmp_path / 'base', job)
    write_json(folder / 'adapter/adapter_config.json', {'r': 16})
    (folder / 'adapter/adapter_model.safetensors').write_bytes(b'fixture')
    write_json(folder / 'training.json', {'job': job, 'status': 'complete',
        'adapter_sha256': adapter_signature(folder / 'adapter')})
    jobs = discover(tmp_path / 'joint', tmp_path / 'registry.json', tmp_path / 'base')
    assert jobs[job['id']][0] == folder.resolve()
