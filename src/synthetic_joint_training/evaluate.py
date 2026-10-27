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



class EvaluationSession:
    """Reuse one tokenizer, prompt cache, and vLLM backbone across adapters."""
    def __init__(self, max_rank=32):
        self.max_rank = max_rank
        self.tokenizer = None
        self.identity = None
        self.llm = None
        self.engine_identity = None
        self.prompts = {}
        self.adapter_number = 0

    def get_tokenizer(self, job):
        key = (job['model'], job['revision'])
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(job['model'], revision=job['revision'])
            self.identity = key
        elif self.identity != key:
            raise ValueError('An evaluation session cannot mix model revisions')
        return self.tokenizer

    def get_engine(self, job, args):
        key = (job['model'], job['revision'], job['recipe']['dtype'], args.tensor_parallel_size,
               args.gpu_memory_utilization, args.max_model_len, args.eval_seed)
        if self.llm is not None:
            if self.engine_identity != key:
                raise ValueError('An evaluation session cannot mix engine settings')
            return self.llm
        from vllm import LLM
        kwargs = {'model': job['model'], 'revision': job['revision'], 'enable_lora': True,
                  'max_loras': 1, 'max_cpu_loras': 1,
                  'max_lora_rank': max(8, 2 ** (self.max_rank - 1).bit_length()),
                  'tensor_parallel_size': args.tensor_parallel_size, 'dtype': job['recipe']['dtype'],
                  'gpu_memory_utilization': args.gpu_memory_utilization, 'seed': args.eval_seed}
        if args.max_model_len:
            kwargs['max_model_len'] = args.max_model_len
        self.llm = LLM(**kwargs)
        self.engine_identity = key
        return self.llm

    def request(self, adapter):
        if adapter is None:
            return None
        from vllm.lora.request import LoRARequest
        self.adapter_number += 1
        return LoRARequest(f'adapter-{self.adapter_number}', self.adapter_number, str(adapter))


def evaluate(job, rows, output, adapter, args, session=None):
    from tqdm.auto import tqdm
    configured_ranks = [args.lora_rank] if args.lora_rank is not None else args.lora_ranks
    session = session or EvaluationSession(max_rank=max(job['recipe']['lora_rank'], *configured_ranks))
    if job['recipe']['lora_rank'] > session.max_rank:
        raise ValueError('Adapter rank exceeds the shared inference capacity')
    output = Path(output)
    selected = [r for r in rows if r['split'] == args.eval_split]
    if not selected:
        raise ValueError(f'No rows for eval split {args.eval_split}')
    tokenizer = session.get_tokenizer(job)
    template_sha = digest(tokenizer.chat_template)
    import importlib.metadata
    from .train import adapter_signature
    adapter_sha = adapter_signature(adapter) if adapter else None
    if adapter:
        state = read_json(Path(adapter).parent / 'training.json')
        if state.get('chat_template_sha256') != template_sha:
            raise ValueError('Evaluation chat template differs from training')
    batch_size = args.eval_batch_size
    if batch_size < 1:
        raise ValueError('Inference batch size must be positive')
    settings = {'versions': {p: importlib.metadata.version(p) for p in ('vllm', 'transformers')},
                'engine_protocol': 'shared-lora-v1', 'eval_batch_size': batch_size,
                'max_lora_rank': max(8, 2 ** (session.max_rank - 1).bit_length()),
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
    report_path = folder / 'report.json'
    if report_path.exists():
        return read_json(report_path)
    from vllm import SamplingParams
    llm = session.get_engine(job, args)
    request = session.request(adapter)
    metrics = {}
    label = ('Original' if adapter is None else
             f"B{job['benchmark_percent']} r{job['recipe']['lora_rank']} s{job['seed']}")
    for style in args.prompt_styles:
        prompt_key = (settings['eval_sha256'], style, template_sha)
        if prompt_key not in session.prompts:
            session.prompts[prompt_key] = [tokenizer.apply_chat_template(benchmark_messages(row, style), tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for row in selected]
        prompts = session.prompts[prompt_key]
        max_tokens = args.direct_max_tokens if style == 'Direct' else args.cot_max_tokens
        predictions = []
        with tqdm(range(0, len(selected), batch_size), position=1, leave=False,
                  desc=f"Inference batches | {job['benchmark']} {label} {style}") as batches:
            for start in batches:
                stop = min(start + batch_size, len(selected))
                # Disable vLLM's request bar: the outer adapter and batch bars
                # provide stable totals without an extra progress layer.
                outputs = llm.generate(prompts[start:stop], SamplingParams(temperature=0.0, max_tokens=max_tokens,
                    seed=args.eval_seed), lora_request=request, use_tqdm=False)
                if len(outputs) != stop - start:
                    raise ValueError('Inference output count mismatch')
                for row, prompt, result in zip(selected[start:stop], prompts[start:stop], outputs):
                    completion = result.outputs[0]
                    predictions.append({'query_id': row['query_id'], 'gold': row['gold'], 'prompt': prompt,
                        'completion': completion.text, 'completion_token_ids': list(completion.token_ids),
                        'generated_tokens': len(completion.token_ids), 'finish_reason': completion.finish_reason,
                        **parse_completion(completion.text, row, style)})
                batches.set_postfix(examples=f'{len(predictions)}/{len(selected)}',
                    accuracy=f"{sum(r['correct'] for r in predictions) / len(predictions):.2%}")
        write_rows(folder / f'{style}.jsonl', predictions)
        metrics[style] = summarize(predictions)
    report = {'schema': 1, 'identity': identity, 'job': job, 'metrics': metrics,
              'predictions': {s: f'{s}.jsonl' for s in args.prompt_styles}}
    write_json(report_path, report)
    return report
