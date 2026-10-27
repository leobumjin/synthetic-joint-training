"""Local adapters and immutable Hugging Face snapshots share one registry."""
import os
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from .io import read_json, write_json, lock
from .train import adapter_signature


def register(registry, job, folder):
    registry = Path(registry)
    with lock(registry.with_suffix('.lock')):
        data = read_json(registry) if registry.exists() else {'schema': 1, 'adapters': {}}
        state = read_json(Path(folder) / 'training.json')
        if state['status'] != 'complete':
            raise ValueError('Cannot register incomplete training')
        existing = data['adapters'].get(job['id'], {})
        if existing and existing['adapter_sha256'] != state['adapter_sha256']:
            raise ValueError('Registry already binds this job to different adapter weights')
        data['adapters'][job['id']] = {**existing, 'job': job, 'adapter_sha256': state['adapter_sha256'],
            'local_path': os.path.relpath(Path(folder).resolve(), registry.parent.resolve())}
        write_json(registry, data)


def registry_entry(registry, job):
    registry = Path(registry)
    data = read_json(registry)
    if job['id'] not in data['adapters']:
        raise ValueError(f"Missing base LoRA {job['id']} ({job['model']}, {job['benchmark']}, "
                         f"B{job['benchmark_percent']}, seed={job['seed']}). Run run_base_lora.sh "
                         'or provide a matching registry. Base training recipe must match.')
    entry = data['adapters'][job['id']]
    if entry['job'] != job:
        raise ValueError('Registry job identity mismatch')
    return entry


def resolve(registry, job):
    registry = Path(registry)
    entry = registry_entry(registry, job)
    if entry.get('repo_id'):
        from huggingface_hub import snapshot_download
        revision = entry.get('revision', '')
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Published adapter revision must be a full immutable commit SHA')
        checkpoint = safe_remote_path(entry.get('checkpoint_path', ''))
        prefix = checkpoint + '/' if checkpoint else ''
        snapshot = Path(snapshot_download(entry['repo_id'], revision=revision,
                        allow_patterns=[prefix + 'adapter/*', prefix + 'training.json']))
        folder = snapshot / checkpoint
    else:
        folder = registry.parent / entry['local_path']
    state = read_json(folder / 'training.json')
    if (state['status'] != 'complete' or state['job'] != job or
            adapter_signature(folder / 'adapter') != entry['adapter_sha256'] or
            state['adapter_sha256'] != entry['adapter_sha256']):
        raise ValueError('Base adapter metadata/content mismatch')
    if entry.get('artifact_sha256') and artifact_signature(folder) != entry['artifact_sha256']:
        raise ValueError('Published checkpoint files changed')
    return folder.resolve()



DEFAULT_BASE_REPO = 'synthetic-joint-training-base-loras'
DEFAULT_SYNTHETIC_REPO = 'synthetic-joint-training-synthetic-loras'


def repository_id(job, namespace, repo_name=None):
    """Shared model repository; experiment identity belongs in its subfolder."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', namespace):
        raise ValueError('HF namespace must be a user or organization name, without slashes')
    name = repo_name or (DEFAULT_SYNTHETIC_REPO if job.get('synthetic_id') else DEFAULT_BASE_REPO)
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) or
            '..' in name or '--' in name or name.endswith(('.', '-'))):
        raise ValueError('HF repo name must be a valid short repository name without a namespace')
    return f'{namespace}/{name}'


def safe_remote_path(value):
    if not isinstance(value, str):
        raise ValueError('Invalid checkpoint path')
    if value and (PurePosixPath(value).is_absolute() or
                  any(part in ('', '.', '..') for part in value.split('/')) or
                  any(char in value for char in ('*', '?', '[', ']', '\\'))):
        raise ValueError('Unsafe checkpoint path')
    return value


def checkpoint_path(job):
    def component(value):
        return re.sub(r'[^A-Za-z0-9_.-]', '_', str(value))
    if not re.fullmatch(r'[0-9a-f]{64}', job['id']):
        raise ValueError('Job ID must be a full SHA256')
    model = '/'.join(component(part) for part in job['model'].split('/'))
    mode = f"b{job['benchmark_percent']}"
    if job.get('synthetic_id'):
        mode += f"-synth{job['synthetic_percent']}"
        if job.get('control_kind'):
            mode += '-repeat'
    return safe_remote_path(f"models/{model}/{component(job['benchmark'])}/seed-{job['seed']}/{mode}/{job['id']}")


def artifact_signature(folder):
    folder = Path(folder)
    files = sorted([folder / 'training.json', *(folder / 'adapter').rglob('*')])
    hashes = {}
    for path in files:
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            hashes[path.relative_to(folder).as_posix()] = h.hexdigest()
    from .io import digest
    return digest(hashes)


def remote_registry(api, repo):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    head = api.repo_info(repo_id=repo, repo_type='model').sha
    try:
        path = hf_hub_download(repo_id=repo, filename='registry.json', repo_type='model', revision=head)
    except EntryNotFoundError:
        return head, {'schema': 1, 'adapters': {}}
    data = read_json(path)
    if data.get('schema') != 1 or not isinstance(data.get('adapters'), dict):
        raise ValueError('Invalid remote registry schema')
    return head, data


def merge_remote_entry(api, repo, entry):
    from huggingface_hub.errors import HfHubHTTPError
    # Optimistic concurrency: reread/merge when another uploader advanced main.
    for _ in range(5):
        head, data = remote_registry(api, repo)
        old = data['adapters'].get(entry['job']['id'])
        if old:
            if (old['artifact_sha256'] != entry['artifact_sha256'] or old['job'] != entry['job'] or
                    old['checkpoint_path'] != entry['checkpoint_path'] or old['repo_id'] != entry['repo_id']):
                raise ValueError('Remote registry binds this run to different files')
            return old
        data['adapters'][entry['job']['id']] = entry
        try:
            api.upload_file(repo_id=repo, repo_type='model', path_in_repo='registry.json',
                path_or_fileobj=(json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode(),
                parent_commit=head, commit_message='Add checkpoint to shared LoRA registry')
            return entry
        except HfHubHTTPError as exc:
            if getattr(exc.response, 'status_code', None) not in (409, 412):
                raise
    raise RuntimeError('Concurrent registry updates exceeded retries; rerun upload to finish indexing')


def verify_local(folder, job):
    folder = Path(folder)
    state = read_json(folder / 'training.json')
    if (state.get('status') != 'complete' or state['job'] != job or
            adapter_signature(folder / 'adapter') != state['adapter_sha256']):
        raise ValueError(f'Incomplete or changed adapter: {folder}')
    return state


def upload_artifact(folder, job, namespace, private=False, repo_name=None):
    from huggingface_hub import HfApi
    state = verify_local(folder, job)
    repo = repository_id(job, namespace, repo_name)
    checkpoint = checkpoint_path(job)
    signature = artifact_signature(folder)
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type='model', private=private, exist_ok=True)
    _, index = remote_registry(api, repo)
    existing = index['adapters'].get(job['id'])
    if existing:
        if (existing['artifact_sha256'] != signature or existing['job'] != job or
                existing['checkpoint_path'] != checkpoint):
            raise ValueError('Remote checkpoint already exists with different files or identity')
        print(f'Skipped unchanged checkpoint: {repo}/{checkpoint}')
        return existing
    commit = api.upload_folder(repo_id=repo, repo_type='model', folder_path=str(folder),
                              path_in_repo=checkpoint, allow_patterns=['adapter/*', 'training.json'],
                              commit_message=f"Add LoRA checkpoint {job['id'][:12]}")
    entry = {'repo_id': repo, 'revision': commit.oid, 'checkpoint_path': checkpoint,
             'subfolder': checkpoint + '/adapter', 'job': job,
             'adapter_sha256': state['adapter_sha256'], 'artifact_sha256': signature}
    entry = merge_remote_entry(api, repo, entry)
    print(f"Published https://huggingface.co/{repo}/tree/{entry['revision']}/{checkpoint}")
    return entry


def publish(registry, job, namespace, private=False, repo_name=None):
    registry = Path(registry)
    entry = read_json(registry)['adapters'][job['id']]
    # Upload local weights directly, including when a previous HF revision exists.
    local = registry.parent / entry['local_path'] if entry.get('local_path') else None
    folder = local if local is not None and local.exists() else resolve(registry, job)
    state = verify_local(folder, job)
    if state['adapter_sha256'] != entry['adapter_sha256']:
        raise ValueError('Adapter no longer matches registry')
    uploaded = upload_artifact(folder, job, namespace, private, repo_name)
    with lock(registry.with_suffix('.lock')):
        data = read_json(registry)
        if data['adapters'][job['id']]['adapter_sha256'] != state['adapter_sha256']:
            raise ValueError('Registry changed during upload')
        data['adapters'][job['id']].update(uploaded)
        write_json(registry, data)
    return uploaded
