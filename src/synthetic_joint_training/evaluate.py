"""Frozen prompts and completions; extraction and requested format are separate."""
from pathlib import Path
from collections import Counter
from .benchmarks import benchmark_messages, parse_completion, PARSER_VERSION
from .io import digest, read_json, write_json, write_rows


def summarize(predictions):
    n = len(predictions)
    return {'n': n, 'accuracy': sum(r['correct'] for r in predictions) / n,
            'extraction_failure_rate': sum(not r['answer_extracted'] for r in predictions) / n,
            'format_violation_rate': sum(not r['requested_format_valid'] for r in predictions) / n,
            'length_limit_rate': sum(r['finish_reason'] == 'length' for r in predictions) / n,
            'mean_generated_tokens': sum(r['generated_tokens'] for r in predictions) / n,
            'extraction_errors': dict(Counter(r.get('extraction_error') or 'unspecified'
                                     for r in predictions if not r['answer_extracted']))}


def evaluate(job, rows, output, adapter, args):
    from transformers import AutoTokenizer
    output = Path(output)
    selected = [r for r in rows if r['split'] == args.eval_split]
    if not selected:
        raise ValueError(f'No rows for eval split {args.eval_split}')
    tokenizer = AutoTokenizer.from_pretrained(job['model'], revision=job['revision'])
    template_sha = digest(tokenizer.chat_template)
    import importlib.metadata
    from .train import adapter_signature
    adapter_sha = adapter_signature(adapter) if adapter else None
    if adapter:
        state = read_json(Path(adapter).parent / 'training.json')
        if state.get('chat_template_sha256') != template_sha:
            raise ValueError('Evaluation chat template differs from training')
    settings = {'versions': {p: importlib.metadata.version(p) for p in ('vllm', 'transformers')},
                'parser': PARSER_VERSION, 'eval_sha256': digest(selected), 'split': args.eval_split,
                'styles': args.prompt_styles, 'direct_max_tokens': args.direct_max_tokens,
                'cot_max_tokens': args.cot_max_tokens, 'temperature': 0.0, 'seed': args.eval_seed,
                'enable_thinking': False, 'chat_template_sha256': template_sha,
                'tensor_parallel_size': args.tensor_parallel_size, 'dtype': job['recipe']['dtype'],
                'gpu_memory_utilization': args.gpu_memory_utilization}
    if args.max_model_len:
        settings['max_model_len'] = args.max_model_len
    identity = {'job_id': job['id'], 'adapter': bool(adapter), 'adapter_sha256': adapter_sha, 'settings': settings}
    folder = output / digest(identity)
    report = folder / 'report.json'
    if report.exists():
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kwargs = {'model': job['model'], 'revision': job['revision'], 'enable_lora': bool(adapter),
              'tensor_parallel_size': args.tensor_parallel_size, 'dtype': job['recipe']['dtype'],
              'gpu_memory_utilization': args.gpu_memory_utilization, 'seed': args.eval_seed}
    if args.max_model_len:
        kwargs['max_model_len'] = args.max_model_len
    if adapter:
        rank = read_json(Path(adapter) / 'adapter_config.json')['r']
        kwargs['max_lora_rank'] = max(8, 2 ** (rank - 1).bit_length())
    llm = LLM(**kwargs)
    request = LoRARequest('evaluation', 1, str(adapter)) if adapter else None
    metrics = {}
    for style in args.prompt_styles:
        prompts = [tokenizer.apply_chat_template(benchmark_messages(row, style), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for row in selected]
        max_tokens = args.direct_max_tokens if style == 'Direct' else args.cot_max_tokens
        # No automatic prompt truncation; an overlength request fails explicitly.
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens,
                               seed=args.eval_seed), lora_request=request)
        if len(outputs) != len(selected):
            raise ValueError('Inference output count mismatch')
        predictions = []
        for row, prompt, result in zip(selected, prompts, outputs):
            completion = result.outputs[0]
            predictions.append({'query_id': row['query_id'], 'gold': row['gold'], 'prompt': prompt,
                'completion': completion.text, 'completion_token_ids': list(completion.token_ids),
                'generated_tokens': len(completion.token_ids), 'finish_reason': completion.finish_reason,
                **parse_completion(completion.text, row, style)})
        write_rows(folder / f'{style}.jsonl', predictions)
        metrics[style] = summarize(predictions)
    write_json(report, {'schema': 1, 'identity': identity, 'job': job, 'metrics': metrics,
                        'predictions': {s: f'{s}.jsonl' for s in args.prompt_styles}})
