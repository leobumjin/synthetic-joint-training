"""Frozen benchmark data, zero-shot prompts and additive training mixtures."""
import hashlib
import json
from pathlib import Path

from .answers import numeric_answer, parse_completion, PARSER_VERSION


SOURCES = {
    'gsm8k': {'repo': 'openai/gsm8k', 'revision': '740312add88f781978c0658806c59bc2815b9866',
              'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet',
              'train_split': 'train', 'eval_split': 'test'},
    'mmlu': {'repo': 'cais/mmlu', 'revision': 'c30699e8356da336a370243923dbaf21066bb9fe',
             'train': 'all/auxiliary_train-00000-of-00001.parquet',
             'validation': 'all/validation-00000-of-00001.parquet',
             'test': 'all/test-00000-of-00001.parquet',
             'train_split': 'auxiliary_train', 'eval_split': 'test'},
    'mmlu_pro': {'repo': 'TIGER-Lab/MMLU-Pro', 'revision': 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
                 'train': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet',
                 'train_split': 'validation', 'eval_split': 'test'},
    'folio': {'repo': 'yale-nlp/FOLIO', 'revision': '4c1f14bf39ca6a6de242bd8f75ef313b4b436818',
              'train': 'folio_v2_train.jsonl', 'test': 'folio_v2_validation.jsonl',
              'train_split': 'train', 'eval_split': 'validation'},
}


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(name, raw, split, index):
    """Keep only the task input and direct answer; never expose reference rationales."""
    if name == 'gsm8k':
        question = raw['question'].strip()
        if '####' not in raw['answer']:
            raise ValueError('GSM8K reference lacks final answer delimiter')
        gold = numeric_answer(raw['answer'].rsplit('####', 1)[1])
        payload = {'question': question}
    elif name == 'mmlu':
        raw = raw.get('train', raw)  # Some official auxiliary files wrap each row.
        options = [x.strip() for x in raw['choices']]
        answer = raw['answer']
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4 or len(options) != 4:
            raise ValueError('Invalid MMLU answer/options')
        gold = chr(65 + answer)
        payload = {'question': raw['question'].strip(), 'options': options}
    elif name == 'mmlu_pro':
        options = [x.strip() for x in raw['options']]
        gold = raw['answer'].strip().upper()
        if not 2 <= len(options) <= 26 or gold not in [chr(65+i) for i in range(len(options))]:
            raise ValueError('Invalid MMLU-Pro answer/options')
        if raw.get('answer_index', ord(gold)-65) != ord(gold)-65:
            raise ValueError('MMLU-Pro answer/index mismatch')
        payload = {'question': raw['question'].strip(), 'options': options}
    elif name == 'folio':
        premises = raw['premises']
        if isinstance(premises, str):
            premises = premises.splitlines()
        payload = {'premises': [p.strip() for p in premises], 'conclusion': raw['conclusion'].strip()}
        labels = {'true': 'True', 'false': 'False', 'unknown': 'Unknown', 'uncertain': 'Unknown'}
        gold = labels[str(raw['label']).lower()]
    else:
        raise ValueError(f'Unknown benchmark: {name}')
    key = content_hash(payload)
    return {'query_id': f'{name}:{split}:{index}', 'graph_id': f'{name}:{split}:' + str(raw.get('story_id', index)),
            'benchmark': name, 'split': split, 'input': payload, 'input_sha256': key, 'gold': gold,
            'category': raw.get('category', raw.get('subject') or 'all')}


def benchmark_messages(row, style='Direct', system_prompt_file=None, answer_format_version='json'):
    name, data = row['benchmark'], row['input']
    if name == 'gsm8k':
        task = data['question']
        answer_format = 'a number without units'
    elif name in ('mmlu', 'mmlu_pro'):
        task = data['question'] + '\n\n' + '\n'.join(f'{chr(65+i)}. {s}' for i, s in enumerate(data['options']))
        answer_format = 'one option letter'
    else:
        task = ('Determine whether the conclusion is True (entailed), False (its negation is entailed), '
                'or Unknown (neither is entailed) from the premises.\n\nPremises:\n' +
                '\n'.join(data['premises']) + '\n\nConclusion:\n' + data['conclusion'])
        answer_format = 'True, False, or Unknown'
    if style == 'Direct':
        task += f'\n\nReturn only the final answer: {answer_format}.'
    elif style == 'CoT':
        task += f'\n\nReason step by step. End with "Final answer: <answer>" where <answer> is {answer_format}.'
    else:
        raise ValueError(f'Unknown benchmark prompt style: {style}')
    if answer_format_version == 'json':
        task = task.rsplit('\n\n', 1)[0]
        instruction = ('Return only a JSON object' if style == 'Direct' else
                       'Reason step by step, then end with a JSON object on its own final line')
        task += f'\n\n{instruction}: {{"answer":"<answer>"}}. <answer> must be {answer_format}. Do not use Markdown fences.'
    result = []
    if system_prompt_file:
        result.append({'role': 'system', 'content': Path(system_prompt_file).read_text()})
    return result + [{'role': 'user', 'content': task}]
