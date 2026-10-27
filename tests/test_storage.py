"""Only verified completed runs discard optimizer checkpoints."""
import pytest

from synthetic_joint_training.io import write_json
from synthetic_joint_training.train import adapter_signature, cleanup_checkpoints


def saved_run(root, run_id='a', status='complete'):
    folder = root / 'runs' / run_id
    job = {'id': run_id, 'synthetic_id': None}
    write_json(folder / 'job.json', job)
    write_json(folder / 'adapter/adapter_config.json', {'r': 16})
    (folder / 'adapter/adapter_model.safetensors').write_bytes(b'fixture')
    write_json(folder / 'adapter/tokenizer_config.json', {'fixture': True})
    write_json(folder / 'training.json', {'job': job, 'status': status,
        'adapter_sha256': adapter_signature(folder / 'adapter')})
    write_json(folder / 'checkpoints/checkpoint-1/optimizer.pt', {'large': True})
    write_json(folder / 'evaluations/fixture/report.json', {'fixture': True})
    return folder, job


def test_cleanup_only_verified_complete_training(tmp_path):
    complete, job = saved_run(tmp_path / 'base')
    cleanup_checkpoints(job, complete)
    assert not (complete / 'checkpoints').exists()
    assert (complete / 'adapter/adapter_model.safetensors').exists()
    partial, job = saved_run(tmp_path / 'base', 'b', 'running')
    cleanup_checkpoints(job, partial)
    assert (partial / 'checkpoints').exists()
    corrupt, job = saved_run(tmp_path / 'base', 'c')
    (corrupt / 'adapter/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        cleanup_checkpoints(job, corrupt)
    assert (corrupt / 'checkpoints').exists()


def test_registered_base_rebinds_only_identical_relocated_artifact(tmp_path):
    import shutil
    from synthetic_joint_training.registry import register, reusable_base
    from synthetic_joint_training.io import read_json
    original, job = saved_run(tmp_path / 'old')
    registry = tmp_path / 'configs/registry.json'
    register(registry, job, original)
    moved = tmp_path / 'new/runs/a'
    moved.parent.mkdir(parents=True)
    shutil.move(original, moved)
    assert reusable_base(registry, job, moved) == moved.resolve()
    assert read_json(registry)['adapters']['a']['local_path'] == '../new/runs/a'
    assert reusable_base(registry, {'id': 'unregistered'}, moved) is None


def test_registered_missing_weights_fail_before_retraining(tmp_path):
    import shutil
    from synthetic_joint_training.registry import register, reusable_base
    original, job = saved_run(tmp_path / 'old')
    registry = tmp_path / 'configs/registry.json'
    register(registry, job, original)
    shutil.rmtree(original)
    replacement, _ = saved_run(tmp_path / 'new')
    (replacement / 'adapter/adapter_model.safetensors').write_bytes(b'new weights')
    from synthetic_joint_training.io import read_json
    state = read_json(replacement / 'training.json')
    state['adapter_sha256'] = adapter_signature(replacement / 'adapter')
    write_json(replacement / 'training.json', state)
    with pytest.raises(ValueError, match='Restore the original registered adapter'):
        reusable_base(registry, job, replacement)
    assert (replacement / 'checkpoints').exists()
