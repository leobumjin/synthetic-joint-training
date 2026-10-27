import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from synthetic_joint_training.io import read_json, write_json
from synthetic_joint_training.registry import repository_id, checkpoint_path, publish, register, resolve, upload_artifact
from synthetic_joint_training.push import main
from synthetic_joint_training.train import adapter_signature


def artifact(root, synthetic=False, complete=True):
    job = {'id': ('b' if synthetic else 'a') * 64, 'model': 'Qwen/Qwen3-4B-Instruct-2507',
           'benchmark': 'gsm8k', 'benchmark_percent': 100, 'synthetic_percent': 5 if synthetic else 0,
           'synthetic_id': 'dataset-hash' if synthetic else None, 'seed': 0}
    folder = root / 'runs' / job['id']
    write_json(folder / 'adapter/adapter_config.json', {'r': 16})
    (folder / 'adapter/adapter_model.safetensors').write_bytes(b'weights')
    write_json(folder / 'training.json', {'status': 'complete' if complete else 'running',
               'job': job, 'adapter_sha256': adapter_signature(folder / 'adapter')})
    return job, folder


def fake_hub(monkeypatch):
    import copy
    import tempfile
    calls, repos = [], {}
    module = ModuleType('huggingface_hub')
    errors = ModuleType('huggingface_hub.errors')
    temporary = tempfile.TemporaryDirectory()
    module._temporary = temporary
    class EntryNotFoundError(Exception):
        pass
    class HfHubHTTPError(Exception):
        def __init__(self, message, response=None):
            super().__init__(message)
            self.response = response
    errors.EntryNotFoundError = EntryNotFoundError
    errors.HfHubHTTPError = HfHubHTTPError
    class API:
        def create_repo(self, **kwargs):
            calls.append(('create', kwargs))
            repos.setdefault(kwargs['repo_id'], {'head': '0'*40, 'index': None, 'snapshots': {}})
        def repo_info(self, repo_id, **kwargs):
            return SimpleNamespace(sha=repos[repo_id]['head'])
        def advance(self, repo_id):
            repo = repos[repo_id]
            sha = f"{int(repo['head'], 16) + 1:040x}"
            repo['head'] = sha
            repo['snapshots'][sha] = copy.deepcopy(repo['index'])
            return sha
        def upload_folder(self, **kwargs):
            calls.append(('upload', kwargs))
            return SimpleNamespace(oid=self.advance(kwargs['repo_id']))
        def upload_file(self, **kwargs):
            calls.append(('index', kwargs))
            repo = repos[kwargs['repo_id']]
            if kwargs['parent_commit'] != repo['head']:
                raise HfHubHTTPError('conflict', SimpleNamespace(status_code=409))
            repo['index'] = json.loads(kwargs['path_or_fileobj'])
            return SimpleNamespace(oid=self.advance(kwargs['repo_id']))
    def download(repo_id, filename, revision, **kwargs):
        assert filename == 'registry.json'
        repo = repos[repo_id]
        index = repo['snapshots'].get(revision, repo['index'])
        if index is None:
            raise EntryNotFoundError(filename)
        path = Path(temporary.name) / repo_id.replace('/', '_') / revision / filename
        write_json(path, index)
        return str(path)
    module.HfApi = API
    module.hf_hub_download = download
    module.snapshot_download = lambda **kw: pytest.fail('Upload must not redownload already published weights')
    module._repos = repos
    monkeypatch.setitem(sys.modules, 'huggingface_hub', module)
    monkeypatch.setitem(sys.modules, 'huggingface_hub.errors', errors)
    return calls


def test_dry_run_filters_and_no_side_effects(tmp_path, capsys, monkeypatch):
    output, registry = tmp_path / 'outputs', tmp_path / 'registry.json'
    base, _ = artifact(output)
    artifact(output, synthetic=True)
    calls = fake_hub(monkeypatch)
    main(['--output-root', str(output), '--registry', str(registry),
          '--hf-namespace', 'my-account', '--dry-run'])
    text = capsys.readouterr().out
    assert repository_id(base, 'my-account') in text
    assert 'Dry run: 1 checkpoint' in text
    assert not registry.exists() and not (output / 'hf_uploads.json').exists()
    assert calls == []


def test_upload_all_records_immutable_destinations(tmp_path, monkeypatch):
    output, registry = tmp_path / 'outputs', tmp_path / 'registry.json'
    base, _ = artifact(output)
    synthetic, _ = artifact(output, synthetic=True)
    calls = fake_hub(monkeypatch)
    main(['--output-root', str(output), '--registry', str(registry),
          '--hf-namespace', 'my-account', '--kind', 'all', '--private'])
    uploads = [kw for name, kw in calls if name == 'upload']
    assert len(uploads) == 2
    assert all(kw['allow_patterns'] == ['adapter/*', 'training.json'] for kw in uploads)
    assert all(kw['repo_type'] == 'model' for kw in uploads)
    assert all(kw['private'] for name, kw in calls if name == 'create')
    entries = read_json(registry)['adapters']
    assert len(entries[base['id']]['revision']) == 40
    assert entries[base['id']]['checkpoint_path'] == checkpoint_path(base)
    assert synthetic['id'] not in entries
    catalog = read_json(output / 'hf_uploads.json')['adapters']
    assert set(catalog) == {base['id'], synthetic['id']}
    assert all(next(iter(entry.values()))['subfolder'].endswith('/adapter') for entry in catalog.values())


def test_corruption_aborts_before_upload(tmp_path, monkeypatch):
    output = tmp_path / 'outputs'
    job, folder = artifact(output)
    (folder / 'adapter/adapter_model.safetensors').write_bytes(b'changed')
    calls = fake_hub(monkeypatch)
    with pytest.raises(ValueError, match='changed adapter'):
        main(['--output-root', str(output), '--registry', str(tmp_path / 'registry.json'),
              '--hf-namespace', 'my-account'])
    assert calls == []


def test_existing_publish_uses_local_weights(tmp_path, monkeypatch):
    output, registry = tmp_path / 'outputs', tmp_path / 'registry.json'
    job, folder = artifact(output)
    register(registry, job, folder)
    data = read_json(registry)
    data['adapters'][job['id']].update(repo_id='old/repo', revision='c'*40)
    write_json(registry, data)
    calls = fake_hub(monkeypatch)
    publish(registry, job, 'new-account')
    assert read_json(registry)['adapters'][job['id']]['repo_id'].startswith('new-account/')
    assert [name for name, _ in calls] == ['create', 'upload', 'index']


def test_incomplete_skipped_and_missing_requested_id_errors(tmp_path, capsys):
    output = tmp_path / 'outputs'
    job, _ = artifact(output, complete=False)
    common = ['--output-root', str(output), '--registry', str(tmp_path / 'registry.json'),
              '--hf-namespace', 'my-account']
    main(common)
    assert 'Nothing uploaded' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*common, '--job-ids', job['id']])


def test_repository_names_and_namespace_validation():
    job = {'model': 'org/' + 'long-model'*20, 'benchmark': 'gsm8k', 'benchmark_percent': 75,
           'seed': 0, 'id': 'a'*64}
    assert len(repository_id(job, 'my-account').split('/')[1]) <= 96
    with pytest.raises(ValueError, match='namespace'):
        repository_id(job, 'account/repo')


def test_incremental_upload_preserves_entries_and_skips_unchanged(tmp_path, monkeypatch):
    output = tmp_path / 'outputs'
    first, folder = artifact(output)
    calls = fake_hub(monkeypatch)
    one = upload_artifact(folder, first, 'my-account')
    again = upload_artifact(folder, first, 'my-account')
    assert again == one
    assert sum(name == 'upload' for name, _ in calls) == 1
    second = {**first, 'id': 'c'*64, 'seed': 1}
    second_folder = output / 'runs' / second['id']
    import shutil
    shutil.copytree(folder, second_folder)
    state = read_json(second_folder / 'training.json')
    state['job'] = second
    write_json(second_folder / 'training.json', state)
    two = upload_artifact(second_folder, second, 'my-account')
    assert one['repo_id'] == two['repo_id']
    assert one['checkpoint_path'] != two['checkpoint_path']
    remote = sys.modules['huggingface_hub']._repos[one['repo_id']]['index']
    assert set(remote['adapters']) == {first['id'], second['id']}
    assert remote['adapters'][first['id']]['revision'] == one['revision']
    assert all(kw['path_in_repo'].startswith('models/') for name, kw in calls if name == 'upload')


def test_remote_checkpoint_conflict_does_not_overwrite(tmp_path, monkeypatch):
    job, folder = artifact(tmp_path)
    calls = fake_hub(monkeypatch)
    upload_artifact(folder, job, 'my-account')
    (folder / 'adapter/tokenizer_config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='different files'):
        upload_artifact(folder, job, 'my-account')
    assert sum(name == 'upload' for name, _ in calls) == 1


def test_download_only_selected_checkpoint_and_pinned_revision(tmp_path, monkeypatch):
    import shutil
    job, folder = artifact(tmp_path / 'outputs')
    registry = tmp_path / 'registry.json'
    calls = fake_hub(monkeypatch)
    uploaded = upload_artifact(folder, job, 'my-account')
    # A remote registry must not contain uploader-local paths.
    remote = sys.modules['huggingface_hub']._repos[uploaded['repo_id']]['index']
    assert 'local_path' not in remote['adapters'][job['id']]
    write_json(registry, remote)
    snapshot = tmp_path / 'snapshot'
    target = snapshot / uploaded['checkpoint_path']
    shutil.copytree(folder, target)
    requests = []
    def download(repo_id, **kwargs):
        requests.append((repo_id, kwargs))
        return str(snapshot)
    monkeypatch.setattr(sys.modules['huggingface_hub'], 'snapshot_download', download)
    assert resolve(registry, job) == target.resolve()
    assert requests == [(uploaded['repo_id'], {'revision': uploaded['revision'],
        'allow_patterns': [uploaded['checkpoint_path'] + '/adapter/*',
                           uploaded['checkpoint_path'] + '/training.json']})]


def test_registry_merge_retries_without_losing_concurrent_entry(tmp_path, monkeypatch):
    job, folder = artifact(tmp_path)
    fake_hub(monkeypatch)
    module = sys.modules['huggingface_hub']
    api_class = module.HfApi
    original = api_class.upload_file
    raced = []
    def concurrent_upload(self, **kwargs):
        if not raced:
            raced.append(True)
            repo = module._repos[kwargs['repo_id']]
            repo['index'] = {'schema': 1, 'adapters': {'other-run': {'artifact_sha256': 'unrelated'}}}
            self.advance(kwargs['repo_id'])
        return original(self, **kwargs)
    monkeypatch.setattr(api_class, 'upload_file', concurrent_upload)
    result = upload_artifact(folder, job, 'my-account')
    assert set(module._repos[result['repo_id']]['index']['adapters']) == {'other-run', job['id']}


@pytest.mark.parametrize('path', ['../other', '/absolute', 'models/*', 'models/../other', 'models//other'])
def test_unsafe_checkpoint_path_rejected(path):
    from synthetic_joint_training.registry import safe_remote_path
    with pytest.raises(ValueError, match='Unsafe checkpoint'):
        safe_remote_path(path)
