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
