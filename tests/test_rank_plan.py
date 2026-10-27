"""The approved experiment is 6 base + 24 joint training jobs."""
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
    assert len(jobs) == 6
    assert Counter(j['benchmark_percent'] for j in jobs) == {100: 6}
    assert {(j['recipe']['lora_rank'], j['recipe']['lora_alpha']) for j in jobs} == {(16, 32), (32, 64)}
    args.command = 'synthetic'
    branches, synthetic_bases = build_plan(args, bundles)
    assert bases == synthetic_bases
    assert len(branches) == 24
    assert len({j['id'] for j in jobs + branches}) == 30
    assert Counter((j['benchmark_percent'], j['synthetic_percent']) for j in branches) == {
        (90, 10): 6, (80, 20): 6, (100, 10): 6, (100, 20): 6}
    for child in branches:
        parent = bases[child['reference_id']]
        assert child['recipe'] == parent['recipe']
        assert child['seed'] == parent['seed']
        assert parent['benchmark_percent'] == 100
        assert child['initialization'] == 'original_model'
        assert 'parent_id' not in child
    for rank in (16, 32):
        for seed in (0, 1, 2):
            selected = [j for j in branches if j['recipe']['lora_rank'] == rank and j['seed'] == seed]
            additions = [j for j in selected if j['mode'] == 'addition']
            assert len({j['reference_id'] for j in additions}) == 1


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
    assert len(jobs) == 3
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


def test_training_budget_defaults_and_override_identity(bundles):
    args = parser().parse_args(['base'])
    assert (args.epochs, args.synthetic_epochs) == (5, None)
    assert (args.batch_size, args.gradient_accumulation) == (8, 2)
    assert args.batch_size * args.gradient_accumulation == 16
    jobs, _ = build_plan(args, bundles)
    assert all(j['recipe']['epochs'] == 5 for j in jobs)
    previous = parser().parse_args(['base', '--epochs', '1', '--batch-size', '1', '--gradient-accumulation', '16'])
    old_jobs, _ = build_plan(previous, bundles)
    assert {j['id'] for j in jobs}.isdisjoint(j['id'] for j in old_jobs)


def test_only_b100_is_pretrained_and_existing_identity_is_preserved(bundles):
    from synthetic_joint_training.plan import make_job, recipe
    args = parser().parse_args(['base'])
    jobs, _ = build_plan(args, bundles)
    assert len(jobs) == 6
    assert {job['benchmark_percent'] for job in jobs} == {100}
    for job in jobs:
        original = make_job(job['model'], job['revision'], job['benchmark'], job['benchmark_id'],
                            job['seed'], 100, recipe(args, job['recipe']['lora_rank']))
        assert job['id'] == original['id']


def test_joint_rows_mix_both_sources_deterministically(bundles):
    from synthetic_joint_training.training_data import select_training_rows
    args = parser().parse_args(['synthetic', '--seeds', '0', '--lora-rank', '16'])
    jobs, _ = build_plan(args, bundles)
    expected = {(90, 10): (90, 10), (80, 20): (80, 20),
                (100, 10): (100, 10), (100, 20): (100, 20)}
    for job in jobs:
        rows, reference = select_training_rows(job, bundles['gsm8k'])
        benchmark, synthetic = expected[job['benchmark_percent'], job['synthetic_percent']]
        assert Counter(row['training_source'] for row in rows) == {'benchmark': benchmark, 'synthetic': synthetic}
        assert len({row['id'] for row in rows}) == len(rows) == job['n_examples']
        assert rows == select_training_rows(job, bundles['gsm8k'])[0]
        assert reference is None
        assert job['recipe']['epochs'] == 5
        assert job['initialization'] == 'original_model'
        control_rows, budget_reference = select_training_rows({**job, 'control_kind': 'benchmark-repeat'}, bundles['gsm8k'])
        assert control_rows == bundles['gsm8k'][2]
        assert budget_reference == rows


def test_joint_override_preserves_reference_and_changes_run_identity(bundles):
    args = parser().parse_args(['synthetic'])
    original, bases = build_plan(args, bundles)
    args.synthetic_epochs = 7
    changed, references = build_plan(args, bundles)
    assert references == bases
    assert {j['id'] for j in original}.isdisjoint(j['id'] for j in changed)
    assert all(j['recipe']['epochs'] == 7 and j['base_recipe']['epochs'] == 5 for j in changed)
