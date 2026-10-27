"""Discover saved manifests without original data paths or active env settings."""
import argparse
from collections import defaultdict
from pathlib import Path
import random
import statistics

from .io import digest, read_json, read_rows


def paired_delta(current, baseline, style, bootstrap_samples=0):
    a = read_rows(current['_folder'] / current['predictions'][style])
    b = read_rows(baseline['_folder'] / baseline['predictions'][style])
    left = {r['query_id']: r for r in a}
    right = {r['query_id']: r for r in b}
    if len(left) != len(a) or len(right) != len(b) or left.keys() != right.keys():
        raise ValueError('Paired comparison requires the same unique query IDs')
    if any(left[k]['gold'] != right[k]['gold'] for k in left):
        raise ValueError('Paired comparison gold labels differ')
    return [float(left[k]['correct']) - float(right[k]['correct']) for k in sorted(left)]


def mean_sd(values):
    if not values:
        return '—'
    mean = statistics.mean(values) * 100
    return f'{mean:.2f}' + (f' ±{statistics.stdev(values)*100:.2f}' if len(values) > 1 else '')


def delta_summary(pairs, style, bootstrap_samples):
    if not pairs:
        return '—'
    diffs = [paired_delta(a, b, style) for a, b in pairs]
    point = statistics.mean(statistics.mean(d) for d in diffs) * 100
    text = f'{point:+.2f} [{len(pairs)} seeds]'
    if bootstrap_samples:
        # Conditional on these training seeds; resample shared evaluation questions.
        if len({len(d) for d in diffs}) != 1:
            raise ValueError('Paired seeds have different evaluation sizes')
        per_question = [statistics.mean(v) for v in zip(*diffs)]
        rng, n = random.Random(42), len(per_question)
        draws = sorted(sum(per_question[rng.randrange(n)] for _ in range(n)) / n * 100
                       for _ in range(bootstrap_samples))
        text += f' CI[{draws[int(.025*(bootstrap_samples-1))]:+.2f},{draws[int(.975*(bootstrap_samples-1))]:+.2f}]'
    return text


def print_table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    print('  '.join(str(x).ljust(w) for x, w in zip(headers, widths)))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(row, widths)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path('outputs'),
                   help='Joint SFT output root (also searched for legacy mixed runs)')
    p.add_argument('--base-output-root', type=Path, default=Path('outputs_base_lora'),
                   help='Base LoRA and reference evaluation output root')
    p.add_argument('--models', nargs='+')
    p.add_argument('--benchmarks', nargs='+')
    p.add_argument('--bootstrap-samples', type=int, default=0,
                   help='Optional paired question bootstrap, conditional on available training seeds')
    p.add_argument('--plain', action='store_true', help='Compatibility option; output is always plain text')
    args = p.parse_args(argv)
    if args.bootstrap_samples < 0:
        p.error('bootstrap-samples must be nonnegative')
    roots = list(dict.fromkeys([args.base_output_root.resolve(), args.output_root.resolve()]))
    reports, seen = [], {}
    paths = {path for root in roots for path in root.glob('runs/*/evaluations/*/report.json')}
    for path in sorted(paths):
        report = read_json(path)
        job = report['job']
        if args.models and job['model'] not in args.models:
            continue
        if args.benchmarks and job['benchmark'] not in args.benchmarks:
            continue
        key = (job['id'], digest(report['identity']))
        if key in seen:
            if seen[key] != report:
                raise ValueError('Conflicting duplicate evaluation reports across output roots')
            continue
        seen[key] = dict(report)
        report['_folder'] = path.parent
        reports.append(report)
    if not reports:
        print('No completed evaluations in ' + ', '.join(str(root) for root in roots))
    # Never combine evaluation protocols, model revisions, or benchmark manifests.
    cohorts = defaultdict(list)
    for report in reports:
        j = report['job']
        key = (j['model'], j['revision'], j['benchmark'], j['benchmark_id'],
               digest(report['identity']['settings']))
        cohorts[key].append(report)
    for key, cohort in cohorts.items():
        model, revision, name, data_id, eval_id = key
        print(f'\n{model} | {name} | revision={revision[:12]} | data={data_id[:12]} | eval={eval_id[:12]}')
        by_id = {r['job']['id']: r for r in cohort}
        styles = cohort[0]['identity']['settings']['styles']
        groups = defaultdict(list)
        for r in cohort:
            j = r['job']
            groups[(j.get('control_kind', j.get('kind', 'trained')), j['benchmark_percent'], j['synthetic_percent'],
                    j['synthetic_id'], digest({'stage': j['recipe'], 'base': j.get('base_recipe', j['recipe']), 'synthetic_name': j.get('synthetic_name'), 'protocol': j.get('protocol'), 'training_mode': j.get('training_mode')}))].append(r)
        for style in styles:
            print(f'\n{style} · % · mean ± sample SD · deltas in percentage points')
            table = []
            for group, items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][3]), kv[0][4])):
                kind, fraction, syn, syn_id, recipe_id = group
                pairs_parent, pairs_full = [], []
                for r in items:
                    j = r['job']
                    parent = by_id.get(j.get('parent_id'))
                    if parent:
                        pairs_parent.append((r, parent))
                    # B100 must have exactly the parent recipe, not synthetic-stage overrides.
                    baseline_recipe = j.get('base_recipe', j['recipe'])
                    full = [b for b in cohort if b['job']['benchmark_percent'] == 100 and
                            not b['job']['synthetic_id'] and b['job']['seed'] == j['seed'] and
                            b['job']['recipe'] == baseline_recipe]
                    if j.get('reference_id'):
                        full = [b for b in full if b['job']['id'] == j['reference_id']]
                    if len(full) == 1:
                        pairs_full.append((r, full[0]))
                metrics = [r['metrics'][style] for r in items]
                label = ('Base model' if kind == 'unadapted' else
                         f'B100 repeat (B{fraction}+S{syn} budget)' if kind == 'benchmark-repeat' else
                         f'B{fraction} + S{syn} joint' if items[0]['job'].get('training_mode') == 'joint' else
                         f'B{fraction}' if not syn_id else f'B{fraction} → S{syn} sequential')
                table.append([label, '—' if kind == 'unadapted' else str(items[0]['job']['recipe']['lora_rank']),
                    '—' if kind == 'unadapted' else str(items[0]['job']['recipe']['lora_alpha']), items[0]['job'].get('synthetic_name') or (syn_id or '-')[:10], recipe_id[:8], str(len(items)),
                    mean_sd([m['accuracy'] for m in metrics]),
                    mean_sd([m['extraction_failure_rate'] for m in metrics]),
                    mean_sd([m['format_violation_rate'] for m in metrics]),
                    mean_sd([m['length_limit_rate'] for m in metrics]),
                    delta_summary(pairs_parent, style, args.bootstrap_samples),
                    delta_summary(pairs_full, style, args.bootstrap_samples)])
            print_table(['Condition', 'Rank', 'Alpha', 'Synthetic', 'Recipe', 'Seeds', 'Accuracy', 'Extract fail',
                         'Format fail', 'Length limit', 'Δ parent', 'Δ B100'], table)
    known = {}
    for path in (path for root in roots for path in root.glob('plans/*.json')):
        for job in read_json(path)['jobs']:
            known[job['id']] = job
    for path in (path for root in roots for path in root.glob('runs/*/job.json')):
        job = read_json(path)
        known[job['id']] = job
    known = {key: job for key, job in known.items() if
             (not args.models or job['model'] in args.models) and
             (not args.benchmarks or job['benchmark'] in args.benchmarks)}
    states = []
    for key in known:
        paths = [root / 'runs' / key / 'training.json' for root in roots]
        saved = [read_json(path).get('status') for path in paths if path.exists()]
        states.append('complete' if 'complete' in saved else 'running' if 'running' in saved else 'pending')
    print(f'\nPlanned/saved training jobs: {len(known)} | complete: {states.count("complete")} | '
          f'partial: {states.count("running")} | pending: {states.count("pending")} | '
          f'completed evaluation reports: {len(reports)}')
    print('Available seeds only; compare paired deltas. Question bootstrap CIs, when requested, '
          'are conditional on these training seeds.')


if __name__ == '__main__':
    main()
