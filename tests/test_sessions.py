"""Shared runtimes must preserve independent adapters and accurate batch totals."""
import importlib.metadata
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from synthetic_joint_training.cli import parser
from synthetic_joint_training.io import digest, read_json, write_json
from synthetic_joint_training.paths import run_folder
from synthetic_joint_training.plan import make_job, recipe
from synthetic_joint_training.train import TrainingSession, train


def job(seed=0, rank=16):
    args = parser().parse_args(['base', '--lora-rank', str(rank)])
    return make_job('test/model', 'a'*40, 'gsm8k', 'data', seed, 100, recipe(args))


def training_runtime(monkeypatch):
    events, rng = [], random.Random()
    class Backbone:
        config = SimpleNamespace(use_cache=True, text_config=SimpleNamespace(use_cache=True))
        weight = 17
        def zero_grad(self, **kwargs):
            events.append('zero_grad')
    base = Backbone()
    class Tokenizer:
        chat_template = 'test-template'
        pad_token_id = 0
        def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
            events.append('encode')
            text = ''.join(f"<{m['role']}>{m['content']}|" for m in messages)
            return list((text + ('<assistant>' if add_generation_prompt else '')).encode())
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            write_json(Path(path) / 'tokenizer_config.json', {'template': self.chat_template})
    tokenizer = Tokenizer()
    class Adapter:
        def __init__(self, backbone, config):
            assert not hasattr(backbone, 'peft_config')
            self.backbone, self.config, self.initial = backbone, config, rng.random()
            self.value = self.initial
            backbone.peft_config = {'default': config}
            events.append(('attach', config.r, self.initial))
        def named_parameters(self):
            return [('weight', SimpleNamespace(requires_grad=False)), ('lora_A', SimpleNamespace(requires_grad=True))]
        def enable_input_require_grads(self):
            pass
        def disable_input_require_grads(self):
            events.append('disable_input_grads')
        def unload(self):
            events.append('unload_without_merge')
            return self.backbone
    class Trainer:
        def __init__(self, **kwargs):
            self.model, self.args, self.callbacks = kwargs['model'], kwargs['args'], kwargs['callbacks']
            self.state = SimpleNamespace(global_step=0, max_steps=2, num_input_tokens_seen=64)
            self.accelerator = SimpleNamespace(free_memory=lambda: events.append('free_optimizer'))
            self.train_dataset = kwargs['train_dataset']
            events.append(('new_optimizer', self.model.initial))
        def remove_callback(self, callback):
            pass
        def train(self, **kwargs):
            callback = self.callbacks[0]
            callback.on_train_begin(self.args, self.state, None)
            for _ in range(2):
                self.model.value += 1
                self.state.global_step += 1
                callback.on_step_end(self.args, self.state, None)
            callback.on_train_end(self.args, self.state, None)
            events.append(('trained', self.model.value))
            return SimpleNamespace(metrics={'loss': .1})
        def save_model(self, path):
            path = Path(path)
            write_json(path / 'adapter_config.json', {'r': self.model.config.r})
            (path / 'adapter_model.safetensors').write_bytes(str(self.model.value).encode())
    def arguments(**kwargs):
        return SimpleNamespace(**kwargs, train_batch_size=kwargs['per_device_train_batch_size'], world_size=1)
    modules = {name: ModuleType(name) for name in ['torch', 'datasets', 'peft', 'transformers',
               'transformers.trainer_callback', 'transformers.trainer_utils']}
    modules['torch'].bfloat16, modules['torch'].float16, modules['torch'].float32 = 'bf16', 'fp16', 'fp32'
    modules['datasets'].Dataset = SimpleNamespace(from_list=lambda rows: rows)
    modules['peft'].LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    modules['peft'].get_peft_model = lambda base, cfg: Adapter(base, cfg)
    modules['peft'].PeftModel = SimpleNamespace(from_pretrained=lambda *a, **kw: pytest.fail('Not a continuation test'))
    transformers = modules['transformers']
    def load_model(*args, **kwargs):
        events.append('load_model')
        return base
    def load_tokenizer(*args, **kwargs):
        events.append('load_tokenizer')
        return tokenizer
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
    transformers.Trainer, transformers.TrainingArguments = Trainer, arguments
    transformers.TrainerCallback = type('TrainerCallback', (), {})
    transformers.set_seed = lambda seed: rng.seed(seed)
    modules['transformers.trainer_callback'].PrinterCallback = type('PrinterCallback', (), {})
    modules['transformers.trainer_utils'].get_last_checkpoint = lambda path: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'fixture')
    return events, base, Trainer


def test_training_reuses_backbone_resets_adapter_and_caches_tokens(tmp_path, monkeypatch):
    events, base, _ = training_runtime(monkeypatch)
    rows = [{'id': 'q', 'messages': [{'role': 'user', 'content': 'Question'},
                                    {'role': 'assistant', 'content': 'Answer'}]}]
    session = TrainingSession()
    first, second = job(0, 16), job(0, 32)
    train(first, rows, tmp_path / 'first', session=session)
    train(second, rows, tmp_path / 'second', session=session)
    assert events.count('load_model') == events.count('load_tokenizer') == 1
    assert events.count('encode') == 2  # Prefix + full template, once for the shared example.
    assert events.count('unload_without_merge') == events.count('free_optimizer') == 2
    starts = [event[2] for event in events if isinstance(event, tuple) and event[0] == 'attach']
    assert starts[0] == starts[1]  # Same seed, reset before independent adapter initialization.
    assert base.weight == 17 and not hasattr(base, 'peft_config')
    assert not base.config.use_cache and not base.config.text_config.use_cache
    assert read_json(tmp_path / 'first/training.json')['status'] == 'complete'
    before = list(events)
    train(first, rows, tmp_path / 'first', session=session)
    assert events == before  # Completed runs do not initialize adapters or load models.


def test_failed_training_unloads_adapter_and_releases_optimizer(tmp_path, monkeypatch):
    events, base, trainer_class = training_runtime(monkeypatch)
    def fail(self, **kwargs):
        raise RuntimeError('fixture training failure')
    monkeypatch.setattr(trainer_class, 'train', fail)
    rows = [{'id': 'q', 'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}]
    session = TrainingSession()
    with pytest.raises(RuntimeError, match='fixture training failure'):
        train(job(), rows, tmp_path, session=session)
    assert 'unload_without_merge' in events and 'free_optimizer' in events
    assert session.model is base and base.weight == 17
    assert read_json(tmp_path / 'training.json')['status'] == 'running'


def test_evaluation_reuses_engine_unique_adapter_ids_and_batches(tmp_path, monkeypatch):
    from synthetic_joint_training.evaluate import EvaluationSession, evaluate
    from synthetic_joint_training.train import adapter_signature
    calls, loads = [], []
    class Tokenizer:
        chat_template = 'template'
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
    transformers = ModuleType('transformers')
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
    vllm, request_module = ModuleType('vllm'), ModuleType('vllm.lora.request')
    class LLM:
        def __init__(self, **kwargs):
            loads.append(kwargs)
        def generate(self, prompts, params, lora_request, use_tqdm):
            assert not use_tqdm
            calls.append((len(prompts), lora_request))
            return [SimpleNamespace(outputs=[SimpleNamespace(text='{"answer":"42"}', token_ids=[1],
                    finish_reason='stop')]) for _ in prompts]
    vllm.LLM, vllm.SamplingParams = LLM, lambda **kwargs: kwargs
    request_module.LoRARequest = lambda name, index, path: (name, index, path)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', request_module)
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'fixture')
    args = parser().parse_args(['base', '--eval-batch-size', '2', '--prompt-styles', 'Direct'])
    rows = [{'benchmark': 'gsm8k', 'input': {'question': f'q{i}'}, 'split': 'test',
             'query_id': f'q{i}', 'gold': '42'} for i in range(5)]
    session = EvaluationSession(max_rank=32)
    for index, rank in enumerate((16, 32)):
        current = job(index, rank)
        folder = tmp_path / str(index)
        write_json(folder / 'adapter/adapter_config.json', {'r': rank})
        (folder / 'adapter/adapter_model.safetensors').write_bytes(b'weights')
        write_json(folder / 'training.json', {'chat_template_sha256': digest('template')})
        report = evaluate(current, rows, folder / 'evaluations', folder / 'adapter', args, session=session)
        assert report['metrics']['Direct']['accuracy'] == 1
    assert len(loads) == 1
    assert loads[0]['max_loras'] == loads[0]['max_cpu_loras'] == 1
    assert [n for n, _ in calls] == [2, 2, 1, 2, 2, 1]
    assert [request[1] for _, request in calls] == [1, 1, 1, 2, 2, 2]
    assert len(session.prompts) == 1


def test_base_worker_keeps_one_training_session_and_skips_completed_jobs(tmp_path, monkeypatch):
    from synthetic_joint_training.base import train_bases
    from synthetic_joint_training.benchmarks import normalize
    from synthetic_joint_training.data import freeze
    from synthetic_joint_training.plan import build_plan
    events, base, _ = training_runtime(monkeypatch)
    args = parser().parse_args(['base', '--models', 'test/model', '--revision', 'a'*40,
                               '--seeds', '0', '--lora-rank', '16', '--data-root', str(tmp_path / 'data'),
                               '--base-output-root', str(tmp_path / 'base'), '--registry', str(tmp_path / 'registry.json')])
    train_rows = [normalize('gsm8k', {'question': f'q{i}', 'answer': f'#### {i}'}, 'train', i) for i in range(20)]
    eval_rows = [normalize('gsm8k', {'question': 'heldout', 'answer': '#### 0'}, 'test', 0)]
    entry = freeze(args.data_root, 'gsm8k', train_rows, eval_rows, {}, 42)
    write_json(args.data_root / 'index.json', {'schema': 1, 'benchmarks': {'gsm8k': entry}})
    from synthetic_joint_training.data import load_bundle
    jobs, _ = build_plan(args, {'gsm8k': load_bundle(args.data_root, 'gsm8k')})
    payload = {'jobs': jobs, 'args': vars(args)}
    train_bases(payload)
    assert events.count('load_model') == 1
    assert events.count('unload_without_merge') == 1
    assert len(read_json(args.registry)['adapters']) == 1
    # Re-running the normal base command cleans legacy recovery checkpoints
    # without loading or retraining the completed adapter.
    checkpoints = run_folder(args.base_output_root, jobs[0]) / 'checkpoints'
    write_json(checkpoints / 'checkpoint-1/optimizer.pt', {'fixture': True})
    # An old hash-only folder is relocated and reused on the normal command.
    legacy = args.base_output_root / 'runs' / jobs[0]['id']
    checkpoints.parent.rename(legacy)
    before = list(events)
    train_bases(payload)
    assert events == before
    assert not legacy.exists()
    assert not checkpoints.exists()
    assert (checkpoints.parent / 'adapter/adapter_model.safetensors').exists()
    assert base.weight == 17

    # A stale index cannot prevent rebuilding a missing run.
    import shutil
    shutil.rmtree(checkpoints.parent)
    train_bases(payload)
    assert events.count('load_model') == 2
    assert (checkpoints.parent / 'adapter/adapter_model.safetensors').exists()


def test_stale_launcher_cannot_train_with_changed_protocol(tmp_path):
    stale = {**job(), 'protocol': 'sequential-direct-json-v1'}
    with pytest.raises(ValueError, match='restart the launcher'):
        train(stale, [], tmp_path)


def test_joint_worker_uses_fresh_adapters_and_one_original_backbone(tmp_path, monkeypatch):
    from synthetic_joint_training.base import train_bases
    from synthetic_joint_training.benchmarks import normalize
    from synthetic_joint_training.data import freeze, load_bundle
    from synthetic_joint_training.plan import build_plan
    events, backbone, _ = training_runtime(monkeypatch)
    args = parser().parse_args(['synthetic', '--models', 'test/model', '--revision', 'a'*40,
        '--seeds', '0', '--lora-rank', '16', '--data-root', str(tmp_path / 'data'),
        '--output-root', str(tmp_path / 'joint'), '--registry', str(tmp_path / 'missing.json')])
    rows = [normalize('gsm8k', {'question': f'q{i}', 'answer': f'#### {i}'}, 'train', i) for i in range(20)]
    evaluation = [normalize('gsm8k', {'question': 'heldout', 'answer': '#### 0'}, 'test', 0)]
    synthetic = [{'prompt': f'synthetic {i}', 'response': str(i)} for i in range(4)]
    entry = freeze(args.data_root, 'gsm8k', rows, evaluation, {}, 42, synthetic)
    write_json(args.data_root / 'index.json', {'schema': 1, 'benchmarks': {'gsm8k': entry}})
    jobs, _ = build_plan(args, {'gsm8k': load_bundle(args.data_root, 'gsm8k', True)})
    train_bases({'jobs': jobs, 'args': vars(args)})
    assert events.count('load_model') == 1
    assert events.count('unload_without_merge') == 4
    starts = [event[2] for event in events if isinstance(event, tuple) and event[0] == 'attach']
    assert len(starts) == 4 and len(set(starts)) == 1
    assert backbone.weight == 17
    assert not args.registry.exists()
    for job in jobs:
        state = read_json(run_folder(args.output_root, job) / 'training.json')
        assert state['status'] == 'complete'
        assert state['job']['initialization'] == 'original_model'


def test_base_worker_uses_current_files_and_reconciles_old_registry(tmp_path, monkeypatch):
    import shutil
    from synthetic_joint_training.base import train_bases
    from synthetic_joint_training.registry import register
    from synthetic_joint_training.train import adapter_signature
    events, _, _ = training_runtime(monkeypatch)
    current = job()
    original = tmp_path / 'old/runs' / current['id']
    train(current, [{'id': 'q', 'messages': [{'role': 'user', 'content': 'q'},
                                           {'role': 'assistant', 'content': 'a'}]}], original)
    registry = tmp_path / 'configs/registry.json'
    register(registry, current, original)
    root = tmp_path / 'new'
    retry = run_folder(root, current)
    shutil.copytree(original, retry)
    (retry / 'adapter/adapter_model.safetensors').write_bytes(b'different retry')
    state = read_json(retry / 'training.json')
    state['adapter_sha256'] = adapter_signature(retry / 'adapter')
    write_json(retry / 'training.json', state)
    write_json(original / 'checkpoints/checkpoint-1/optimizer.pt', {})
    write_json(retry / 'checkpoints/checkpoint-1/optimizer.pt', {})
    args = parser().parse_args(['base', '--base-output-root', str(root), '--registry', str(registry)])
    before = list(events)
    train_bases({'jobs': [current], 'args': vars(args)})
    assert events == before  # No tokenizer, backbone, optimizer, or training.
    assert (original / 'checkpoints').exists()
    assert not (retry / 'checkpoints').exists()
    assert (retry / 'adapter/adapter_model.safetensors').read_bytes() == b'different retry'
    assert read_json(registry)['adapters'][current['id']]['adapter_sha256'] == state['adapter_sha256']
