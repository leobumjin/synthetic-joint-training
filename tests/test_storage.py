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


def test_base_registration_reconciles_stale_index_and_preserves_history(tmp_path):
    from synthetic_joint_training.registry import register
    from synthetic_joint_training.io import read_json
    folder, job = saved_run(tmp_path / 'base')
    registry = tmp_path / 'configs/registry.json'
    register(registry, job, folder)
    old = read_json(registry)['adapters']['a']
    old.update(repo_id='fixture/published', revision='a'*40)
    write_json(registry, {'schema': 1, 'adapters': {'a': old, 'unrelated': {'fixture': True}}})
    (folder / 'adapter/adapter_model.safetensors').write_bytes(b'retrained')
    state = read_json(folder / 'training.json')
    state['adapter_sha256'] = adapter_signature(folder / 'adapter')
    write_json(folder / 'training.json', state)
    with pytest.raises(ValueError, match='different adapter weights'):
        register(registry, job, folder)
    register(registry, job, folder, reconcile=True)
    current = read_json(registry)['adapters']
    assert current['a']['adapter_sha256'] == state['adapter_sha256']
    assert 'repo_id' not in current['a']
    assert current['unrelated'] == {'fixture': True}
    assert list(read_json(registry.with_name('registry.history.json'))['entries'].values()) == [old]
