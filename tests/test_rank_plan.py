"""The approved experiment is 18 base + 24 continuation training jobs."""
from collections import Counter

import pytest

from synthetic_joint_training.cli import parser, validate
from synthetic_joint_training.data import freeze, load_bundle
from synthetic_joint_training.io import write_json
from synthetic_joint_training.plan import build_plan


@pytest.fixture
def bundles(tmp_path):
    from synthetic_joint_training.benchmarks import normalize
    train = [normalize('gsm8k', {'question': f'q{i}', 'answer': f'#### {i}'}, 'train', i) for i in range(100)]
    evaluation = [normalize('gsm8k', {'question': 'heldout', 'answer': '#### 0'}, 'test', 0)]
    synthetic = [{'prompt': f'synthetic {i}', 'response': str(i)} for i in range(20)]
    entry = freeze(tmp_path, 'gsm8k', train, evaluation, {}, 42, synthetic)
    write_json(tmp_path / 'index.json', {'schema': 1, 'benchmarks': {'gsm8k': entry}})
    return {'gsm8k': load_bundle(tmp_path, 'gsm8k', True)}


def test_approved_grid_counts_parentage_and_scaling(bundles):
    args = parser().parse_args(['base'])
    jobs, bases = build_plan(args, bundles)
    assert len(jobs) == 18
    assert Counter(j['benchmark_percent'] for j in jobs) == {80: 6, 90: 6, 100: 6}
    assert {(j['recipe']['lora_rank'], j['recipe']['lora_alpha']) for j in jobs} == {(16, 32), (32, 64)}
    args.command = 'synthetic'
    branches, synthetic_bases = build_plan(args, bundles)
    assert bases == synthetic_bases
    assert len(branches) == 24
    assert len({j['id'] for j in jobs + branches}) == 42
    assert Counter((j['benchmark_percent'], j['synthetic_percent']) for j in branches) == {
        (90, 10): 6, (80, 20): 6, (100, 10): 6, (100, 20): 6}
    for child in branches:
        parent = bases[child['parent_id']]
        assert child['recipe'] == parent['recipe']
        assert child['seed'] == parent['seed']
        assert child['benchmark_percent'] == parent['benchmark_percent']
    for rank in (16, 32):
        for seed in (0, 1, 2):
            selected = [j for j in branches if j['recipe']['lora_rank'] == rank and j['seed'] == seed]
            additions = [j for j in selected if j['mode'] == 'addition']
            assert len({j['parent_id'] for j in additions}) == 1


@pytest.mark.parametrize('mode,expected', [('replacement', {(90, 10), (80, 20)}),
                                          ('addition', {(100, 10), (100, 20)})])
def test_two_conditions_per_mode(bundles, mode, expected):
    args = parser().parse_args(['synthetic', '--mode', mode])
    jobs, _ = build_plan(args, bundles)
    assert len(jobs) == 12
    assert {(j['benchmark_percent'], j['synthetic_percent']) for j in jobs} == expected


def test_single_rank_override_and_explicit_alpha(bundles):
    args = parser().parse_args(['base', '--lora-ranks', '16', '32', '--lora-rank', '32'])
    jobs, _ = build_plan(args, bundles)
    assert len(jobs) == 9
    assert all(j['recipe']['lora_rank'] == 32 and j['recipe']['lora_alpha'] == 64 for j in jobs)
    args.lora_alpha = 32
    changed, _ = build_plan(args, bundles)
    assert {j['id'] for j in jobs}.isdisjoint(j['id'] for j in changed)


@pytest.mark.parametrize('flags', [['--lora-ranks', '16', '16'], ['--lora-ranks', '0'],
                                  ['--lora-rank', '-1'], ['--lora-alpha', '0']])
def test_invalid_rank_configuration(flags):
    p = parser()
    with pytest.raises(SystemExit):
        validate(p, p.parse_args(['base', *flags]))


def test_output_defaults_are_separate_and_not_in_run_identity(bundles):
    p = parser()
    original = p.parse_args(['base'])
    moved = p.parse_args(['base', '--output-root', '/tmp/synthetic', '--base-output-root', '/tmp/base'])
    assert str(original.output_root) == 'outputs'
    assert str(original.base_output_root) == 'outputs_base_lora'
    assert build_plan(original, bundles) == build_plan(moved, bundles)
