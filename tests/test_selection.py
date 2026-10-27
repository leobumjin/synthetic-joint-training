import json
import subprocess

import pytest

from synthetic_joint_training.benchmarks import normalize
from synthetic_joint_training.cli import main, parser
from synthetic_joint_training.data import custom_split, freeze, load_bundle
from synthetic_joint_training.io import read_json, write_json, write_rows


def rows(n, split='train'):
    return [normalize('gsm8k', {'question': f'{split} question {i}?', 'answer': f'#### {i}'}, split, i)
            for i in range(n)]


def test_custom_split_seed_caps_and_official_pooling():
    pool = rows(100) + rows(100, 'test')
    train, test, info = custom_split(pool, 42, .25, 40, 120)
    assert len(train) == 40 and len(test) == 120
    assert info['unused_unique_inputs'] == 40
    assert {r['input_sha256'] for r in train}.isdisjoint(r['input_sha256'] for r in test)
    assert train == custom_split(pool[::-1], 42, .25, 40, 120)[0]
    assert train != custom_split(pool, 43, .25, 40, 120)[0]
    assert all(r['split'] == 'train' for r in train)
    assert all(r['split'] == 'test' for r in test)
    assert {'train', 'test'} == {r['source_split'] for r in train}


def test_custom_split_preserves_groups_and_deduplicates():
    pool = rows(80)
    for i, row in enumerate(pool):
        row['group_id'] = f'group-{i // 4}'
    pool.append(dict(pool[0], query_id='duplicate'))
    train, test, info = custom_split(pool, 42)
    assert len(train) == 20 and len(test) == 60
    assert info['unique_inputs'] == 80
    assert {r['group_id'] for r in train}.isdisjoint(r['group_id'] for r in test)
    assert custom_split(pool, 42) == custom_split(pool, 42)


def test_conflicting_labels_rejected():
    pool = rows(40)
    pool.append(dict(pool[0], query_id='conflict', gold='999'))
    with pytest.raises(ValueError, match='Conflicting gold'):
        custom_split(pool, 42)


def test_named_datasets_share_base_and_get_separate_plans(tmp_path, capsys):
    root = tmp_path / 'data'
    train, evaluation = rows(100), rows(4, 'test')
    edit = [{'prompt': f'edit {i}', 'response': str(i)} for i in range(30)]
    rebuild = [{'prompt': f'rebuild {i}', 'response': str(i)} for i in range(30)]
    first = freeze(root, 'gsm8k', train, evaluation, {'fixture': True}, 42, edit)
    second = freeze(root, 'gsm8k', train, evaluation, {'fixture': True}, 42, rebuild)
    assert first['benchmark_id'] == second['benchmark_id']
    write_json(root / 'index.json', {'schema': 1, 'benchmarks': {'gsm8k': {
        'benchmark_id': first['benchmark_id'], 'synthetic_id': None,
        'synthetics': {'edit': first['synthetic_id'], 'rebuild': second['synthetic_id']}}}})
    assert load_bundle(root, 'gsm8k', True, 'edit')[0]['synthetic_id'] == first['synthetic_id']
    common = ['--data-root', str(root), '--seeds', '0', '--dry-run']
    main(['synthetic', *common])
    output = capsys.readouterr().out
    decoder, plans = json.JSONDecoder(), []
    while output.strip():
        plan, end = decoder.raw_decode(output.lstrip())
        plans.append(plan)
        output = output.lstrip()[end:]
    assert [p['synthetic'] for p in plans] == ['edit', 'rebuild']
    assert all(len(p['jobs']) == 8 for p in plans)
    assert {j['parent_id'] for j in plans[0]['jobs']} == {j['parent_id'] for j in plans[1]['jobs']}
    assert {j['id'] for j in plans[0]['jobs']}.isdisjoint(j['id'] for j in plans[1]['jobs'])
    main(['synthetic', *common, '--synthetic', 'edit=/unused/source/edit.jsonl'])
    assert json.loads(capsys.readouterr().out)['synthetic'] == 'edit'
    assert parser().parse_args(['base']).prompt_styles == ['Direct', 'CoT']


def test_prepare_script_named_pools_and_default_custom_split(tmp_path):
    source = tmp_path / 'normalized/gsm8k'
    write_rows(source / 'train.jsonl', rows(100))
    write_rows(source / 'eval.jsonl', rows(100, 'test'))
    edit, rebuild = tmp_path / 'edit.jsonl', tmp_path / 'rebuild.jsonl'
    write_rows(edit, [{'prompt': f'edit {i}', 'response': str(i)} for i in range(30)])
    write_rows(rebuild, [{'prompt': f'rebuild {i}', 'response': str(i)} for i in range(30)])
    root = tmp_path / 'prepared'
    command = ['bash', 'shells/prepare_dataset.sh', '--data-root', str(root),
               '--local-benchmark-root', str(source.parent), '--synthetic', f'edit={edit}', f'rebuild={rebuild}']
    subprocess.run(command, check=True, text=True, capture_output=True)
    index = read_json(root / 'index.json')['benchmarks']['gsm8k']
    assert set(index['synthetics']) == {'edit', 'rebuild'}
    _, manifest, train, evaluation, synthetic = load_bundle(root, 'gsm8k', True, 'edit')
    assert manifest['n_train'] == 50 and manifest['n_eval'] == 150
    assert len(synthetic) == 30
    # Preparing only benchmarks must retain existing synthetic selections.
    subprocess.run(command[:command.index('--synthetic')], check=True, text=True, capture_output=True)
    assert read_json(root / 'index.json')['benchmarks']['gsm8k']['synthetics'] == index['synthetics']
