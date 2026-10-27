"""Frozen benchmark subsets and named user-provided synthetic pools."""
import argparse
import json
import random
import re
from pathlib import Path

from .benchmarks import SOURCES, normalize, benchmark_messages
from .io import digest, read_json, read_rows, write_json, write_rows

FRACTIONS = (80, 90, 100)
RATIOS = (10, 20)


def count(n, percent):
    return n * percent // 100


def synthetic_count(n, benchmark_percent, synthetic_percent):
    # Replacement complements the benchmark subset exactly, including rounding.
    return n - count(n, benchmark_percent) if benchmark_percent < 100 else count(n, synthetic_percent)


def ordered(rows, seed):
    rows = sorted(rows, key=lambda r: r['id'])
    random.Random(seed).shuffle(rows)
    return rows


def validate_messages(messages):
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError('Each training row requires user/assistant messages')
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {'role', 'content'} or
                message['role'] not in ('system', 'user', 'assistant') or
                not isinstance(message['content'], str) or not message['content'].strip()):
            raise ValueError('Messages must contain only nonempty role/content strings')
    body = messages[1:] if messages[0]['role'] == 'system' else messages
    if len(body) % 2 or any(m['role'] != ('user' if i % 2 == 0 else 'assistant')
                            for i, m in enumerate(body)):
        raise ValueError('Messages must alternate user/assistant, ending in assistant')


def normalize_synthetic(rows):
    result, ids, content = [], set(), set()
    for raw in rows:
        messages = raw.get('messages')
        if messages is None:
            messages = [{'role': 'user', 'content': raw.get('prompt')},
                        {'role': 'assistant', 'content': raw.get('response')}]
        validate_messages(messages)
        key = digest(messages)
        identifier = raw.get('id', key)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Synthetic id must be a nonempty string')
        if identifier in ids or key in content:
            raise ValueError(f'Duplicate synthetic id/content: {identifier}')
        ids.add(identifier)
        content.add(key)
        row = {'id': identifier, 'messages': messages}
        if 'group_id' in raw:
            if not isinstance(raw['group_id'], str) or not raw['group_id']:
                raise ValueError('group_id must be a nonempty string')
            row['group_id'] = raw['group_id']
        result.append(row)
    if not result:
        raise ValueError('Synthetic pool is empty')
    return result


def training_row(row):
    return {'id': row['query_id'], 'messages': benchmark_messages(row, 'Direct') +
            [{'role': 'assistant', 'content': json.dumps({'answer': row['gold']}, separators=(',', ':'))}]}


def freeze(root, name, train, evaluation, provenance, seed, synthetic=None):
    if not train or not evaluation:
        raise ValueError('Benchmark train and evaluation must be nonempty')
    if len({r['query_id'] for r in train + evaluation}) != len(train + evaluation):
        raise ValueError('Duplicate benchmark IDs')
    heldout = {r['input_sha256'] for r in evaluation}
    if heldout & {r['input_sha256'] for r in train}:
        raise ValueError('Exact benchmark train/evaluation input overlap')
    if len({r['input_sha256'] for r in train}) != len(train):
        raise ValueError('Duplicate benchmark training inputs')
    examples = ordered([training_row(r) for r in train], seed)
    # All official heldout splits are retained; selection happens only at evaluation.
    identity = {'schema': 1, 'benchmark': name, 'data_seed': seed,
                'train_sha256': digest(examples), 'eval_sha256': digest(evaluation),
                'source': provenance, 'supervision': 'direct_json_answer_only'}
    folder = Path(root) / 'benchmarks' / digest(identity)
    write_rows(folder / 'train.jsonl', examples)
    write_rows(folder / 'eval.jsonl', evaluation)
    write_json(folder / 'manifest.json', {**identity, 'id': digest(identity), 'n_train': len(examples),
               'n_eval': len(evaluation), 'subsets': {str(p): count(len(examples), p) for p in FRACTIONS}})
    if synthetic is not None:
        pool = ordered(normalize_synthetic(synthetic), seed)
        if count(len(examples), min(RATIOS)) < 1:
            raise ValueError('Benchmark needs at least 10 rows for 10% experiments')
        needed = len(examples) - count(len(examples), min(FRACTIONS))
        if len(pool) < needed:
            raise ValueError(f'Synthetic pool needs at least {needed} unique rows')
        # Detect exact prompt overlaps with heldout; no claim of semantic decontamination.
        eval_prompts = {digest(benchmark_messages(r, s)) for r in evaluation for s in ('Direct', 'CoT')}
        for row in pool:
            if digest(row['messages'][:-1]) in eval_prompts:
                raise ValueError('Synthetic prompt overlaps benchmark heldout')
        syn_identity = {'schema': 1, 'data_seed': seed, 'rows_sha256': digest(pool)}
        syn_folder = Path(root) / 'synthetic' / digest(syn_identity)
        write_rows(syn_folder / 'train.jsonl', pool)
        write_json(syn_folder / 'manifest.json', {**syn_identity, 'id': digest(syn_identity), 'n_rows': len(pool)})
        syn_id = digest(syn_identity)
    else:
        syn_id = None
    return {'benchmark_id': digest(identity), 'synthetic_id': syn_id}


def synthetic_name(spec):
    name = spec.split('=', 1)[0] if '=' in spec else (Path(spec).stem if '/' in spec or spec.endswith('.jsonl') else spec)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError(f'Invalid synthetic dataset name: {name!r}')
    return name


def load_bundle(root, name, require_synthetic=False, synthetic=None):
    index = read_json(Path(root) / 'index.json')['benchmarks'][name]
    index = dict(index)
    if synthetic is not None:
        if synthetic not in index.get('synthetics', {}):
            raise ValueError(f'Synthetic dataset {synthetic!r} is not prepared for {name}')
        index['synthetic_id'] = index['synthetics'][synthetic]
    elif require_synthetic and not index.get('synthetic_id'):
        choices = index.get('synthetics', {})
        if len(choices) == 1:
            index['synthetic_id'] = next(iter(choices.values()))
        elif len(choices) > 1:
            raise ValueError('Multiple synthetic datasets are prepared; select a name')
    folder = Path(root) / 'benchmarks' / index['benchmark_id']
    manifest = read_json(folder / 'manifest.json')
    train, evaluation = read_rows(folder / 'train.jsonl'), read_rows(folder / 'eval.jsonl')
    if digest(train) != manifest['train_sha256'] or digest(evaluation) != manifest['eval_sha256']:
        raise ValueError('Benchmark content checksum mismatch; prepare again')
    identity = {k: manifest[k] for k in ('schema', 'benchmark', 'data_seed', 'train_sha256',
                                        'eval_sha256', 'source', 'supervision')}
    if digest(identity) != index['benchmark_id'] or manifest['id'] != index['benchmark_id']:
        raise ValueError('Benchmark manifest identity mismatch')
    pool = None
    if require_synthetic:
        if not index.get('synthetic_id'):
            raise ValueError('Prepare synthetic datasets first with --synthetic name=/path/train.jsonl')
        syn_folder = Path(root) / 'synthetic' / index['synthetic_id']
        syn = read_json(syn_folder / 'manifest.json')
        pool = read_rows(syn_folder / 'train.jsonl')
        if digest(pool) != syn['rows_sha256'] or digest({k: syn[k] for k in
                ('schema', 'data_seed', 'rows_sha256')}) != index['synthetic_id']:
            raise ValueError('Synthetic content/manifest checksum mismatch')
    return index, manifest, train, evaluation, pool


def custom_split(rows, seed, train_fraction=.25, max_train=1000, max_test=3000):
    """Pool source splits, deduplicate inputs, and assign whole groups before caps."""
    def group_key(row):
        return row.get('group_id', row.get('graph_id', row['input_sha256']))
    # Duplicate inputs can connect source groups; union them before deduplication.
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    unique = {}
    for row in sorted(rows, key=lambda r: r['query_id']):
        key = row['input_sha256']
        group = find(group_key(row))
        if key in unique:
            if unique[key]['gold'] != row['gold']:
                raise ValueError('Conflicting gold labels for the same benchmark input')
            parents[group] = find(group_key(unique[key]))
        unique.setdefault(key, row)
    groups = {}
    for row in unique.values():
        groups.setdefault(find(group_key(row)), []).append(row)
    units = [groups[key] for key in sorted(groups)]
    random.Random(seed).shuffle(units)
    target = int(len(unique) * train_fraction)
    train_groups, test_groups, allocated = [], [], 0
    for unit in units:
        if allocated + len(unit) <= target:
            train_groups.append(unit)
            allocated += len(unit)
        else:
            test_groups.append(unit)
    def capped(units, limit, split):
        result = []
        for unit in units:
            if len(result) + len(unit) <= limit:
                result.extend({**row, 'source_split': row.get('source_split', row['split']), 'split': split}
                              for row in unit)
        return result
    train, test = capped(train_groups, max_train, 'train'), capped(test_groups, max_test, 'test')
    if not train or not test:
        raise ValueError('Custom split is empty; increase caps or provide more independent groups')
    return train, test, {'policy': 'pooled_unique_group_split_v1', 'seed': seed,
        'train_fraction': train_fraction, 'max_train': max_train, 'max_test': max_test,
        'source_rows': len(rows), 'unique_inputs': len(unique), 'groups': len(units),
        'selected_train': len(train), 'selected_test': len(test),
        'unused_unique_inputs': len(unique) - len(train) - len(test),
        'selected_train_ids_sha256': digest([r['query_id'] for r in train]),
        'selected_test_ids_sha256': digest([r['query_id'] for r in test])}


def download_benchmark(name, cache, mmlu_samples, seed, split_policy='custom',
                       train_fraction=.25, max_train=1000, max_test=3000):
    from huggingface_hub import hf_hub_download
    source = SOURCES[name]
    normalized = {}
    for split in ('train', 'validation', 'test') if name == 'mmlu' else ('train', 'test'):
        path = Path(hf_hub_download(source['repo'], source[split], repo_type='dataset',
                                   revision=source['revision'], cache_dir=cache))
        if path.suffix == '.parquet':
            import pyarrow.parquet as pq
            raw = pq.read_table(path).to_pylist()
        else:
            raw = read_rows(path)
        normalized[split] = [normalize(name, row, split, i) for i, row in enumerate(raw)]
        if name == 'folio':
            for row, original in zip(normalized[split], raw):
                if 'story_id' in original:
                    row['group_id'] = f"folio:story:{original['story_id']}"
    if split_policy == 'custom':
        train, evaluation, selection = custom_split(
            [r for rows in normalized.values() for r in rows], seed, train_fraction, max_train, max_test)
        return train, evaluation, {'official': source, 'selection': selection}
    evaluation = [r for split, rows in normalized.items() if split != 'train' for r in rows]
    heldout = {r['input_sha256'] for r in evaluation}
    seen, train = set(), []
    for row in normalized['train']:
        key = row['input_sha256']
        if key not in seen and key not in heldout:
            seen.add(key)
            train.append(row)
    if name == 'mmlu':
        if mmlu_samples > len(train):
            raise ValueError('Requested MMLU sample count exceeds eligible rows')
        train = random.Random(seed).sample(train, mmlu_samples)
    return train, evaluation, {'official': source, 'mmlu_samples': mmlu_samples if name == 'mmlu' else None,
                              'excluded_train_rows': len(normalized['train']) - len(train)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--benchmarks', nargs='+', choices=('gsm8k', 'mmlu', 'folio'), default=['gsm8k'])
    group = p.add_mutually_exclusive_group()
    group.add_argument('--synthetic-file', type=Path, help='Legacy single synthetic pool')
    group.add_argument('--synthetic', nargs='*', help='Named pools: edit=/path/edit.jsonl rebuild=/path/rebuild.jsonl')
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--mmlu-train-samples', type=int, default=3000)
    p.add_argument('--split-policy', choices=('custom', 'official'), default='custom')
    p.add_argument('--train-fraction', type=float, default=.25)
    p.add_argument('--max-train', type=int, default=1000)
    p.add_argument('--max-test', type=int, default=3000)
    p.add_argument('--cache-dir')
    p.add_argument('--local-benchmark-root', type=Path, help='Offline normalized <name>/{train,eval}.jsonl')
    args = p.parse_args()
    if args.data_seed < 0 or args.mmlu_train_samples < 1:
        p.error('Seed must be nonnegative; MMLU sample count must be positive')
    if not 0 < args.train_fraction < 1 or args.max_train < 1 or args.max_test < 1:
        p.error('Train fraction must be between 0 and 1; split caps must be positive')
    synthetic = read_rows(args.synthetic_file) if args.synthetic_file else None
    named = {}
    for spec in args.synthetic or []:
        name = synthetic_name(spec)
        if name in named:
            p.error(f'Duplicate synthetic name: {name}')
        named[name] = read_rows(spec.split('=', 1)[-1])
    index_path = args.data_root / 'index.json'
    index = read_json(index_path) if index_path.exists() else {'schema': 1, 'benchmarks': {}}
    for name in args.benchmarks:
        if args.local_benchmark_root:
            folder = args.local_benchmark_root / name
            train, evaluation = read_rows(folder / 'train.jsonl'), read_rows(folder / 'eval.jsonl')
            provenance = {'local_normalized': True}
            if args.split_policy == 'custom':
                train, evaluation, selection = custom_split(train + evaluation, args.data_seed,
                    args.train_fraction, args.max_train, args.max_test)
                provenance['selection'] = selection
        else:
            train, evaluation, provenance = download_benchmark(name, args.cache_dir, args.mmlu_train_samples, args.data_seed,
                args.split_policy, args.train_fraction, args.max_train, args.max_test)
        entry = freeze(args.data_root, name, train, evaluation, provenance, args.data_seed, synthetic)
        previous = index['benchmarks'].get(name, {})
        pools = dict(previous.get('synthetics', {})) if previous.get('benchmark_id') == entry['benchmark_id'] else {}
        for label, rows in named.items():
            frozen = freeze(args.data_root, name, train, evaluation, provenance, args.data_seed, rows)
            pools[label] = frozen['synthetic_id']
        if pools:
            entry['synthetics'] = pools
        if not args.synthetic_file and len(pools) == 1:
            entry['synthetic_id'] = next(iter(pools.values()))
        elif not args.synthetic_file and not named and previous.get('benchmark_id') == entry['benchmark_id']:
            entry['synthetic_id'] = previous.get('synthetic_id')
        index['benchmarks'][name] = entry
        print(name, json.dumps(entry))
    write_json(index_path, index)


if __name__ == '__main__':
    main()
