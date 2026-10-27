"""Create tiny offline fixtures for planning/tests, not research evaluation."""
import argparse
import json
from pathlib import Path


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/demo'))
    args = p.parse_args()
    for split, count in [('train', 100), ('test', 8)]:
        rows = []
        for i in range(count):
            question = f'{split} demo: what is {i} + 1?'
            import hashlib
            rows.append({'query_id': f'demo:{split}:{i}', 'benchmark': 'gsm8k', 'split': split,
                'input': {'question': question}, 'input_sha256': hashlib.sha256(question.encode()).hexdigest(),
                'gold': str(i + 1)})
        write(args.root / 'normalized/gsm8k' / f'{"train" if split == "train" else "eval"}.jsonl', rows)
    write(args.root / 'synthetic.jsonl', [{'id': f'demo-s{i}', 'prompt': f'Reverse this string: demo{i}',
                                         'response': f'demo{i}'[::-1]} for i in range(30)])
    print(f'Demo fixtures written to {args.root}; do not report their scores as benchmark results.')


if __name__ == '__main__':
    main()
