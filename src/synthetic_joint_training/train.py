"""Assistant-only SFT; explicit adapter continuation with fresh optimizer per stage."""
from pathlib import Path
import importlib.metadata
import gc
import shutil

from .data import validate_messages
from .io import read_json, write_json, digest


def encode_example(tokenizer, row, max_length):
    validate_messages(row['messages'])
    messages = row['messages']
    kwargs = {'tokenize': True, 'enable_thinking': False}
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    if full[:len(prefix)] != prefix:
        raise ValueError(f"Chat template is not prefix-stable for {row['id']}; assistant mask cannot be proven")
    if len(full) > max_length:
        raise ValueError(f"{row['id']}: {len(full)} tokens exceeds max_length={max_length}; no silent truncation")
    if len(full) <= len(prefix):
        raise ValueError(f"{row['id']}: no supervised tokens")
    return {'input_ids': full, 'attention_mask': [1] * len(full),
            'labels': [-100] * len(prefix) + full[len(prefix):]}


def adapter_signature(folder):
    import hashlib
    folder = Path(folder)
    files = sorted([*folder.glob('adapter_model*.safetensors'), *folder.glob('adapter_model*.bin')])
    if not (folder / 'adapter_config.json').exists() or not files:
        raise ValueError(f'Incomplete adapter: {folder}')
    result = {}
    for path in [folder / 'adapter_config.json', *files]:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        result[path.name] = h.hexdigest()
    return digest(result)


def match_repeat_budget(tokenizer, pool, reference, max_length):
    """Nearest full-example input-token budget; never claim exact compute matching."""
    target = sum(len(encode_example(tokenizer, row, max_length)['input_ids']) for row in reference)
    sizes = [len(encode_example(tokenizer, row, max_length)['input_ids']) for row in pool]
    selected, used, i = [], 0, 0
    while used < target:
        row, size = pool[i % len(pool)], sizes[i % len(pool)]
        if selected and abs(used - target) <= abs(used + size - target):
            break
        selected.append({**row, 'id': f"repeat:{i}:{row['id']}"})
        used += size
        i += 1
    return selected, {'reference_input_tokens_per_epoch': target,
                      'control_input_tokens_per_epoch': used, 'input_token_budget_difference': used - target,
                      'policy': 'nearest_whole_example_input_tokens'}



class TrainingSession:
    """Keep pristine frozen model weights resident while replacing one adapter."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.identity = None
        self.encoded = {}

    def load(self, job):
        key = (job['model'], job['revision'], job['recipe']['dtype'])
        if self.model is not None:
            if key != self.identity:
                raise ValueError('A training session cannot mix model revisions or dtypes')
            return self.model, self.tokenizer
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(job['model'], revision=job['revision'])
        if not self.tokenizer.chat_template:
            raise ValueError('Model requires an explicit chat template')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[key[2]]
        self.model = AutoModelForCausalLM.from_pretrained(job['model'], revision=job['revision'], torch_dtype=dtype)
        for config in (self.model.config, getattr(self.model.config, 'text_config', None)):
            if config is not None:
                config.use_cache = False
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        self.identity = key
        return self.model, self.tokenizer

    def encode(self, rows, max_length):
        result = []
        for row in rows:
            key = (digest(row), max_length)
            if key not in self.encoded:
                self.encoded[key] = encode_example(self.tokenizer, row, max_length)
            result.append(self.encoded[key])
        return result

    def release(self, adapter):
        # NEVER merge: the next independent LoRA must see the original weights.
        if hasattr(adapter, 'gradient_checkpointing_disable'):
            adapter.gradient_checkpointing_disable()
        if hasattr(adapter, '_require_grads_hook') or getattr(adapter, '_require_grads_hooks', None):
            adapter.disable_input_require_grads()
        self.model = adapter.unload()
        # PEFT leaves wrapper metadata on the returned backbone in some releases.
        if hasattr(self.model, 'peft_config'):
            delattr(self.model, 'peft_config')
        self.model.zero_grad(set_to_none=True)
        gc.collect()


def completed(job, output):
    path = Path(output) / 'training.json'
    if not path.exists():
        return False
    saved = read_json(path)
    if saved['job'] != job:
        raise ValueError('Training identity changed in existing output')
    if saved.get('status') != 'complete':
        return False
    if adapter_signature(Path(output) / 'adapter') != saved['adapter_sha256']:
        raise ValueError('Completed adapter content changed')
    return True


def cleanup_checkpoints(job, output):
    """Remove resume state only after final adapter integrity is verified."""
    output = Path(output)
    if completed(job, output):
        checkpoints = output / 'checkpoints'
        if checkpoints.exists():
            if checkpoints.is_symlink():
                raise ValueError('Refusing to remove a symlinked checkpoint directory')
            shutil.rmtree(checkpoints)


def train(job, rows, output, parent=None, resume=False, reference=None, session=None):
    cfg = job['recipe']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if completed(job, output):
        cleanup_checkpoints(job, output)
        return
    from .plan import PROTOCOL, JOINT_PROTOCOL
    if job.get('training_mode') == 'joint' and parent is not None:
        raise ValueError('Joint SFT must start from the original model, not a benchmark LoRA')
    if job.get('protocol') not in (PROTOCOL, JOINT_PROTOCOL):
        raise ValueError('Training protocol changed; restart the launcher to create a fresh plan')
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
    from transformers.trainer_callback import PrinterCallback
    from transformers.trainer_utils import get_last_checkpoint
    from tqdm.auto import tqdm
    checkpoint_dir = output / 'checkpoints'
    previous = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if previous and not resume:
        raise ValueError('Partial training exists; use --resume')
    session = session or TrainingSession()
    base, tokenizer = session.load(job)
    budget_info = None
    if reference is not None:
        rows, budget_info = match_repeat_budget(tokenizer, rows, reference, cfg['max_length'])
    encoded = session.encode(rows, cfg['max_length'])
    token_stats = {'examples_per_epoch': len(rows),
                   'input_tokens_per_epoch': sum(len(r['input_ids']) for r in encoded),
                   'supervised_tokens_per_epoch': sum(sum(t != -100 for t in r['labels']) for r in encoded)}
    # Reset immediately before each adapter initialization; cache hits must not
    # change the random sequence used by another seed/condition.
    set_seed(job['seed'])
    if parent:
        parent_state = read_json(Path(parent).parent / 'training.json')
        if (parent_state['job']['id'] != job['parent_id'] or parent_state['status'] != 'complete' or
                adapter_signature(parent) != parent_state['adapter_sha256'] or
                parent_state['chat_template_sha256'] != digest(tokenizer.chat_template)):
            raise ValueError('Parent adapter identity, content, or chat template mismatch')
        model = PeftModel.from_pretrained(base, str(parent), is_trainable=True)
    else:
        targets = cfg['target_modules'] if cfg['target_modules'] == 'all-linear' else cfg['target_modules'].split(',')
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=cfg['lora_rank'],
                   lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'], target_modules=targets))
    trainer, progress = None, None
    try:
        if any(parameter.requires_grad and 'lora_' not in name for name, parameter in model.named_parameters()):
            raise ValueError('Shared training requires all backbone parameters to remain frozen')
        model.enable_input_require_grads()
        state = {'status': 'running', 'job': job, 'tokens': token_stats,
                 'chat_template_sha256': digest(tokenizer.chat_template),
                 'training_row_ids': [r['id'] for r in rows], 'training_rows_sha256': digest(rows),
                 'budget_match': budget_info,
                 'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'datasets')}}
        write_json(output / 'training.json', state)

        def collate(features):
            width = max(len(f['input_ids']) for f in features)
            return {key: torch.tensor([f[key] + [pad] * (width - len(f[key])) for f in features])
                    for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

        class StepProgress(TrainerCallback):
            bar = None
            def on_train_begin(self, args, state, control, **kwargs):
                self.bar = tqdm(total=state.max_steps, initial=state.global_step, position=1, leave=False,
                    desc=f"Steps | {job['benchmark']} B{job['benchmark_percent']} r{cfg['lora_rank']} s{job['seed']}")
                self.bar.set_postfix(examples=len(rows), epochs=cfg['epochs'], micro_batch=cfg['batch_size'],
                                     effective_batch=args.train_batch_size * args.world_size * cfg['gradient_accumulation'])
            def on_step_end(self, args, state, control, **kwargs):
                self.bar.update(state.global_step - self.bar.n)
            def on_log(self, args, state, control, logs=None, **kwargs):
                if self.bar is not None and logs and 'loss' in logs:
                    self.bar.set_postfix(loss=f"{logs['loss']:.4f}")
            def on_train_end(self, args, state, control, **kwargs):
                if self.bar is not None:
                    self.bar.close()
        progress = StepProgress()
        trainer = Trainer(model=model, args=TrainingArguments(
            output_dir=str(checkpoint_dir), num_train_epochs=cfg['epochs'],
            per_device_train_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['gradient_accumulation'],
            learning_rate=cfg['learning_rate'], lr_scheduler_type='constant_with_warmup',
            warmup_ratio=cfg['warmup_ratio'], weight_decay=cfg['weight_decay'],
            bf16=cfg['dtype'] == 'bfloat16', fp16=cfg['dtype'] == 'float16',
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            save_strategy='steps', save_steps=100, save_total_limit=2,
            logging_steps=10, report_to='none', seed=job['seed'], data_seed=job['seed'],
            disable_tqdm=True, remove_unused_columns=False, include_num_input_tokens_seen=True),
            train_dataset=Dataset.from_list(encoded), data_collator=collate, processing_class=tokenizer,
            callbacks=[progress])
        trainer.remove_callback(PrinterCallback)
        result = trainer.train(resume_from_checkpoint=previous if resume else None)
        trainer.save_model(str(output / 'adapter'))
        tokenizer.save_pretrained(output / 'adapter')
        state.update(status='complete', metrics=result.metrics, global_step=trainer.state.global_step,
                     adapter_sha256=adapter_signature(output / 'adapter'),
                     observed_input_tokens=trainer.state.num_input_tokens_seen)
        write_json(output / 'training.json', state)
        cleanup_checkpoints(job, output)
    finally:
        if progress is not None and progress.bar is not None:
            progress.bar.close()
        if trainer is not None:
            trainer.accelerator.free_memory()
            del trainer
        session.release(model)
