import json
from pathlib import Path
import subprocess
import sys

import pytest

from synthetic_joint_training.benchmarks import normalize, parse_completion
from synthetic_joint_training.data import freeze, load_bundle, count, normalize_synthetic
from synthetic_joint_training.io import read_json, read_rows, write_json, write_rows
from synthetic_joint_training.cli import parser, main
from synthetic_joint_training.plan import build_plan
from synthetic_joint_training.registry import register, resolve
from synthetic_joint_training.train import encode_example, adapter_signature
from synthetic_joint_training.results import main as results_main

SHA = 'a' * 40


@pytest.fixture
def prepared(tmp_path):
    train = [normalize('gsm8k', {'question': f'Train {i}?', 'answer': f'#### {i}'}, 'train', i)
             for i in range(100)]
    test = [normalize('gsm8k', {'question': f'Eval {i}?', 'answer': f'#### {i}'}, 'test', i)
            for i in range(4)]
    syn = [{'id': f's{i:03}', 'prompt': f'Synthetic {i}?', 'response': str(i)} for i in range(30)]
    root = tmp_path / 'data'
    index = freeze(root, 'gsm8k', train, test, {'fixture': True}, 42, syn)
    write_json(root / 'index.json', {'schema': 1, 'benchmarks': {'gsm8k': index}})
    return root, tmp_path / 'outputs', tmp_path / 'registry.json'


def arguments(prepared, command='base', *extra):
    root, output, registry = prepared
    return parser().parse_args([command, '--models', 'test/model', '--revision', SHA,
        '--seeds', '0', '--data-root', str(root), '--output-root', str(output),
        '--registry', str(registry), *extra])


def test_nested_splits_and_independent_forks(prepared):
    args = arguments(prepared)
    bundle = load_bundle(args.data_root, 'gsm8k', True)
    jobs, bases = build_plan(args, {'gsm8k': bundle})
    assert [j['n_examples'] for j in jobs] == [100, 100]
    args.command = 'synthetic'
    branches, _ = build_plan(args, {'gsm8k': bundle})
    assert len(branches) == 8
    assert len({j['id'] for j in branches}) == 8
    assert all(j['reference_id'] in bases for j in branches)
    assert len({j['reference_id'] for j in branches if j['mode'] == 'addition'}) == 2
    assert all(j['n_examples'] == 100
               for j in branches if j['mode'] == 'replacement')
    rows = bundle[2]
    assert rows[:80] == rows[:90][:80]


def test_content_identity_survives_move(prepared, tmp_path):
    root, _, _ = prepared
    original = load_bundle(root, 'gsm8k', True)
    moved = tmp_path / 'moved'
    root.rename(moved)
    assert load_bundle(moved, 'gsm8k', True) == original


def test_corruption_detected(prepared):
    root, _, _ = prepared
    index = read_json(root / 'index.json')['benchmarks']['gsm8k']
    path = root / 'benchmarks' / index['benchmark_id'] / 'train.jsonl'
    rows = read_rows(path)
    rows[0]['messages'][-1]['content'] = 'corrupted'
    write_rows(path, rows)
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(root, 'gsm8k')


def test_synthetic_manifest_corruption(prepared):
    root, _, _ = prepared
    index = read_json(root / 'index.json')['benchmarks']['gsm8k']
    path = root / 'synthetic' / index['synthetic_id'] / 'manifest.json'
    manifest = read_json(path)
    manifest['data_seed'] += 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(root, 'gsm8k', True)


@pytest.mark.parametrize('rows', [
    [{'prompt': 'q', 'response': 'a'}, {'prompt': 'q', 'response': 'a'}],
    [{'messages': [{'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'q'}]}],
    [{'prompt': '', 'response': 'a'}],
    [{'id': 3, 'prompt': 'q', 'response': 'a'}],
])
def test_invalid_synthetic_rejected(rows):
    with pytest.raises(ValueError):
        normalize_synthetic(rows)


def test_missing_parent_fails_before_worker(prepared, monkeypatch):
    root, output, registry = prepared
    write_json(registry, {'schema': 1, 'adapters': {}})
    monkeypatch.setattr('synthetic_joint_training.cli.spawn', lambda *a: pytest.fail('Worker should not start'))
    with pytest.raises(ValueError, match='Missing base LoRA'):
        main(['synthetic', '--models', 'test/model', '--revision', SHA, '--seeds', '0',
              '--data-root', str(root), '--output-root', str(output), '--registry', str(registry)])


def test_registry_checks_adapter_bytes(prepared):
    args = arguments(prepared)
    job = build_plan(args, {'gsm8k': load_bundle(args.data_root, 'gsm8k')})[0][0]
    folder = args.output_root / 'runs' / job['id']
    write_json(folder / 'adapter/adapter_config.json', {'r': 16})
    (folder / 'adapter/adapter_model.safetensors').write_bytes(b'test weights')
    signature = adapter_signature(folder / 'adapter')
    write_json(folder / 'training.json', {'status': 'complete', 'job': job, 'adapter_sha256': signature})
    register(args.registry, job, folder)
    assert resolve(args.registry, job) == folder.resolve()
    (folder / 'adapter/adapter_model.safetensors').write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='mismatch'):
        resolve(args.registry, job)


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        text = ''.join(f"<{m['role']}>{m['content']}|" for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(text.encode())


def test_mask_and_overlength_rejection():
    row = {'id': 'a', 'messages': [{'role': 'user', 'content': 'secret question'},
                                  {'role': 'assistant', 'content': 'answer'}]}
    encoded = encode_example(Tokenizer(), row, 1000)
    unmasked = bytes(x for x in encoded['labels'] if x != -100).decode()
    assert unmasked == 'answer|'
    with pytest.raises(ValueError, match='no silent truncation'):
        encode_example(Tokenizer(), row, 4)


def test_conservative_answer_scoring():
    row = normalize('gsm8k', {'question': 'q', 'answer': '#### 42'}, 'test', 0)
    assert parse_completion('{"answer":"42"}', row, 'Direct')['correct']
    assert not parse_completion('Numbers 17 and 42', row, 'CoT')['answer_extracted']
    scored = parse_completion('42', row, 'CoT')
    assert scored['correct'] and not scored['requested_format_valid']
    assert not parse_completion('{"answer":"42"}\n{"answer":"43"}', row, 'CoT')['answer_extracted']


def test_shell_dry_run_both_modes(prepared):
    root, output, registry = prepared
    for script, expected in [('run_base_lora.sh', 2), ('run.sh', 8)]:
        result = subprocess.run(['bash', 'shells/' + script, '--dry-run', '--models', 'test/model',
            '--revision', SHA, '--seeds', '0', '--data-root', str(root), '--output-root', str(output),
            '--registry', str(registry)], text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        assert len(plan['jobs']) == expected
    assert not output.exists()


def test_orchestrator_continuation_and_reporting(prepared, monkeypatch, capsys):
    root, output, registry = prepared
    captured, phases = [], []

    def worker(action, payload, interpreter, folder):
        if action in ('train-base', 'eval-base', 'train-joint', 'eval-joint'):
            phases.append(action)
            selected = payload['jobs'] if action.startswith('train-') else payload.get('originals', []) + payload.get('references', []) + payload['jobs']
            for entry in selected:
                phase_root = Path(payload['args']['output_root' if entry['synthetic_id'] else 'base_output_root'])
                destination = phase_root / 'runs' / entry['id']
                worker('train' if action.startswith('train-') else 'eval', {'job': entry}, interpreter, destination)
                if action == 'train-base':
                    register(Path(payload['args']['registry']), entry, destination)
            return
        job = payload['job']
        captured.append((action, job, payload.get('parent')))
        if action == 'train':
            write_json(folder / 'adapter/adapter_config.json', {'r': 16})
            (folder / 'adapter/adapter_model.safetensors').write_bytes(job['id'].encode())
            write_json(folder / 'training.json', {'status': 'complete', 'job': job,
                'adapter_sha256': adapter_signature(folder / 'adapter')})
        else:
            settings = {'styles': ['Direct'], 'split': 'test'}
            report_folder = folder / 'evaluations' / 'fixture'
            predictions = [{'query_id': f'q{i}', 'gold': '0', 'correct': i < (3 if job['synthetic_id'] else 2)}
                           for i in range(4)]
            write_rows(report_folder / 'Direct.jsonl', predictions)
            write_json(report_folder / 'report.json', {'job': job, 'identity': {'settings': settings},
                'predictions': {'Direct': 'Direct.jsonl'}, 'metrics': {'Direct': {'accuracy': .75 if job['synthetic_id'] else .5,
                'extraction_failure_rate': 0, 'format_violation_rate': 0, 'length_limit_rate': 0}}})

    monkeypatch.setattr('synthetic_joint_training.cli.spawn', worker)
    common = ['--models', 'test/model', '--revision', SHA, '--seeds', '0', '--data-root', str(root),
              '--output-root', str(output), '--base-output-root', str(output.parent / 'outputs_base_lora'),
              '--registry', str(registry)]
    main(['base', *common])
    assert phases == ['train-base', 'eval-base']
    assert [action for action, _, _ in captured] == ['train'] * 2 + ['eval'] * 3
    captured.clear()
    main(['synthetic', *common, '--synthetic-learning-rate', '1e-5'])
    training = [(j, parent) for action, j, parent in captured if action == 'train']
    assert len(training) == 8
    base_output = output.parent / 'outputs_base_lora'
    assert len(list(base_output.glob('runs/*/training.json'))) == 2
    assert len(list(output.glob('runs/*/training.json'))) == 8
    assert all(read_json(path)['job']['synthetic_id'] is None for path in base_output.glob('runs/*/training.json'))
    assert all(read_json(path)['job']['synthetic_id'] for path in output.glob('runs/*/training.json'))
    assert all(parent is None for _, parent in training)
    assert all(j['initialization'] == 'original_model' for j, _ in training)
    assert all(j['recipe']['learning_rate'] == 1e-5 and j['base_recipe']['learning_rate'] == 2e-5 for j, _ in training)
    assert len({j['reference_id'] for j, _ in training if j['mode'] == 'addition'}) == 2
    results_main(['--output-root', str(output), '--base-output-root', str(output.parent / 'outputs_base_lora'),
                  '--bootstrap-samples', '20'])
    text = capsys.readouterr().out
    assert '+25.00' in text and 'Δ B100' in text and 'CI[' in text
    assert 'complete: 10' in text
    # A copied legacy base tree must not double-count seed results or progress.
    import shutil
    shutil.copytree(base_output / 'runs', output / 'runs', dirs_exist_ok=True)
    results_main(['--output-root', str(output), '--base-output-root', str(base_output)])
    copied = capsys.readouterr().out
    assert 'complete: 10' in copied
    assert 'completed evaluation reports: 11' in copied


def test_replacement_rounding_exact_total():
    from synthetic_joint_training.data import synthetic_count
    for n in (20, 21, 99, 101, 1319):
        for fraction in (75, 80, 85, 90, 95):
            assert count(n, fraction) + synthetic_count(n, fraction, 100-fraction) == n
        assert synthetic_count(n, 100, 5) == count(n, 5)


def test_repeat_control_matches_nearest_whole_example():
    from synthetic_joint_training.train import match_repeat_budget
    pool = normalize_synthetic([{'id': 'a', 'prompt': 'q', 'response': 'a'}])
    reference = normalize_synthetic([{'id': f'r{i}', 'prompt': 'q', 'response': 'a'} for i in range(1)])
    rows, stats = match_repeat_budget(Tokenizer(), pool, reference, 1000)
    assert len(rows) == 1
    assert stats['input_token_budget_difference'] == 0


def test_control_plan_adds_two_branches_per_rank(prepared, capsys):
    root, output, registry = prepared
    main(['synthetic', '--models', 'test/model', '--revision', SHA, '--seeds', '0',
          '--data-root', str(root), '--output-root', str(output), '--registry', str(registry),
          '--control', 'benchmark-repeat', '--dry-run'])
    jobs = json.loads(capsys.readouterr().out)['jobs']
    assert len(jobs) == 12
    assert sum(j.get('control_kind') == 'benchmark-repeat' for j in jobs) == 4
    assert len({j['id'] for j in jobs}) == 12


def test_worker_rejects_synthetic_index_change(prepared, monkeypatch):
    from synthetic_joint_training import worker
    root, output, registry = prepared
    args = arguments(prepared, 'synthetic')
    jobs, _ = build_plan(args, {'gsm8k': load_bundle(root, 'gsm8k', True)})
    job = jobs[0]
    job['synthetic_id'] = 'wrong-id'
    request = output / 'request.json'
    write_json(request, {'job': job, 'args': {'data_root': str(root)}})
    monkeypatch.setattr(sys, 'argv', ['worker', 'train', str(request)])
    with pytest.raises(ValueError, match='Synthetic dataset index changed'):
        worker.main()


def test_evaluation_freezes_outputs_and_reuses_completed_report(prepared, monkeypatch):
    import importlib.metadata
    from types import ModuleType, SimpleNamespace
    from synthetic_joint_training.evaluate import evaluate
    args = arguments(prepared)
    bundle = load_bundle(args.data_root, 'gsm8k')
    job = build_plan(args, {'gsm8k': bundle})[0][0]
    args.prompt_styles = ['Direct', 'CoT']
    creations = []

    class EvalTokenizer:
        chat_template = 'fixture-template'

        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

    fake_transformers = ModuleType('transformers')
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: EvalTokenizer())
    fake_vllm = ModuleType('vllm')

    class LLM:
        def __init__(self, **kwargs):
            creations.append(kwargs)

        def generate(self, prompts, params, lora_request, use_tqdm=False):
            return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps({'answer': str(i)}),
                    token_ids=[1, 2], finish_reason='stop')]) for i in range(len(prompts))]

    fake_vllm.LLM = LLM
    fake_vllm.SamplingParams = lambda **kw: kw
    fake_request = ModuleType('vllm.lora.request')
    fake_request.LoRARequest = lambda *a: a
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setitem(sys.modules, 'vllm', fake_vllm)
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', fake_request)
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'fixture')
    folder = args.output_root / 'evaluation-smoke'
    evaluate(job, bundle[3], folder, None, args)
    reports = list(folder.glob('*/report.json'))
    assert len(reports) == 1
    report = read_json(reports[0])
    assert report['metrics']['Direct']['accuracy'] == 1
    assert len(read_rows(reports[0].parent / 'CoT.jsonl')) == 4
    evaluate(job, bundle[3], folder, None, args)
    assert len(creations) == 1  # Cached report never initializes a GPU engine again.


def test_joint_train_only_needs_no_b100_registry(prepared, monkeypatch):
    root, output, registry = prepared
    captured = []
    monkeypatch.setattr('synthetic_joint_training.cli.spawn', lambda action, payload, *unused: captured.append((action, payload)))
    main(['synthetic', '--stage', 'train', '--models', 'test/model', '--revision', SHA,
          '--seeds', '0', '--data-root', str(root), '--output-root', str(output),
          '--registry', str(registry)])
    assert not registry.exists()
    assert len(captured) == 1 and captured[0][0] == 'train-joint'
    assert all(j['initialization'] == 'original_model' and 'parent_id' not in j for j in captured[0][1]['jobs'])
