"""Assistant-only SFT; explicit adapter continuation with fresh optimizer per stage."""
from pathlib import Path
import importlib.metadata

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


def train(job, rows, output, parent=None, resume=False, reference=None):
    cfg = job['recipe']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training.json').exists():
        saved = read_json(output / 'training.json')
        if saved['job'] != job:
            raise ValueError('Training identity changed in existing output')
        if saved.get('status') == 'complete':
            if adapter_signature(output / 'adapter') != saved['adapter_sha256']:
                raise ValueError('Completed adapter content changed')
            return
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(job['seed'])
    tokenizer = AutoTokenizer.from_pretrained(job['model'], revision=job['revision'])
    if not tokenizer.chat_template:
        raise ValueError('Model requires an explicit chat template')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    budget_info = None
    if reference is not None:
        rows, budget_info = match_repeat_budget(tokenizer, rows, reference, cfg['max_length'])
    encoded = [encode_example(tokenizer, row, cfg['max_length']) for row in rows]
    token_stats = {'examples_per_epoch': len(rows),
                   'input_tokens_per_epoch': sum(len(r['input_ids']) for r in encoded),
                   'supervised_tokens_per_epoch': sum(sum(t != -100 for t in r['labels']) for r in encoded)}
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[cfg['dtype']]
    model = AutoModelForCausalLM.from_pretrained(job['model'], revision=job['revision'], torch_dtype=dtype)
    model.config.use_cache = False
    if parent:
        parent_state = read_json(Path(parent).parent / 'training.json')
        if (parent_state['job']['id'] != job['parent_id'] or parent_state['status'] != 'complete' or
                adapter_signature(parent) != parent_state['adapter_sha256'] or
                parent_state['chat_template_sha256'] != digest(tokenizer.chat_template)):
            raise ValueError('Parent adapter identity, content, or chat template mismatch')
        model = PeftModel.from_pretrained(model, str(parent), is_trainable=True)
    else:
        targets = cfg['target_modules'] if cfg['target_modules'] == 'all-linear' else cfg['target_modules'].split(',')
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=cfg['lora_rank'],
                   lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'], target_modules=targets))
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

    checkpoint_dir = output / 'checkpoints'
    previous = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if previous and not resume:
        raise ValueError('Partial training exists; use --resume')
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(checkpoint_dir), num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['gradient_accumulation'],
        learning_rate=cfg['learning_rate'], lr_scheduler_type='constant_with_warmup',
        warmup_ratio=cfg['warmup_ratio'], weight_decay=cfg['weight_decay'],
        bf16=cfg['dtype'] == 'bfloat16', fp16=cfg['dtype'] == 'float16',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='steps', save_steps=100, save_total_limit=2,
        logging_steps=10, report_to='none', seed=job['seed'], data_seed=job['seed'],
        remove_unused_columns=False, include_num_input_tokens_seen=True),
        train_dataset=Dataset.from_list(encoded), data_collator=collate, processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=previous if resume else None)
    trainer.save_model(str(output / 'adapter'))
    tokenizer.save_pretrained(output / 'adapter')
    state.update(status='complete', metrics=result.metrics, global_step=trainer.state.global_step,
                 adapter_sha256=adapter_signature(output / 'adapter'),
                 observed_input_tokens=trainer.state.num_input_tokens_seen)
    write_json(output / 'training.json', state)
