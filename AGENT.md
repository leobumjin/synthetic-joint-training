# Repository guidance

## Purpose and language

This repository evaluates benchmark-only B100 SFT against independent joint benchmark/synthetic SFT from the original model. It is a standalone project; do not introduce runtime dependencies on sibling repositories.

Use English for repository documentation, code comments, docstrings, CLI messages, configuration explanations, and examples. Keep the requested filename `AGENT.md` and the existing upload entry point `hf_push_lora_parmas.sh`.

## Layout

- `shells/env.sh`: central selections and defaults for models, benchmarks, synthetic pools, ranks, seeds, training, and evaluation.
- `shells/_common.sh`: shared CLI forwarding; public entry points should remain thin wrappers.
- `src/synthetic_joint_training/data.py`: source loading, custom splitting, named synthetic pools, immutable data bundles, and validation.
- `plan.py`: deterministic experiment planning without GPU dependencies.
- `cli.py`: orchestration, reference validation, and isolated worker execution.
- `base.py`: model-scoped base training/evaluation phases, backbone reuse, and nested progress.
- `train.py`: final-assistant-only SFT, fresh adapters, checkpoint recovery, and training metadata.
- `evaluate.py`: frozen prompts, generation, predictions, and report metadata.
- `answers.py`: numeric normalization, answer extraction, and format compliance. `benchmarks.py` re-exports the parser interface.
- `results.py`: manifest discovery, separate cohorts, seed statistics, and paired comparisons.
- `registry.py` and `push.py`: local/Hugging Face adapter lookup, selective downloads, incremental publication, and remote registry merging.
- `io.py`: content identities, atomic writes, and local locks.
- `configs/base_lora_registry.json`: initially empty adapter registry, populated by actual base training or publication.
- `tests/`: offline regressions for experiment planning, data integrity, orchestration, parsing, and mocked Hub operations.
- `examples/`: offline demo fixtures, not research datasets.

## Experiment invariants

Keep code, shell defaults, tests, and README descriptions consistent when changing the experiment protocol.

### Conditions and counts

- Base fraction: B100 only; B80/B90 are data subsets in joint mixtures.
- Joint replacement: B90 + S10 and B80 + S20.
- Joint addition: B100 + S10 and B100 + S20.
- Default ranks: 16 and 32, with alpha 32 and 64 respectively.
- Default training seeds: 0, 1, 2.
- For one model and benchmark: 6 base training jobs.
- For each synthetic dataset: 24 joint training jobs, sharing B100 evaluation references.
- One synthetic dataset therefore requires 30 trained artifacts. Direct/CoT evaluation does not create additional training jobs.
- An explicit single `--lora-rank` overrides the configured rank array. An explicit alpha overrides automatic scaling.

### Data and randomness

- Data selection seed and training seeds serve different purposes. Keep data subsets fixed across training seeds.
- Default custom preparation pools official source splits, deduplicates inputs, preserves groups, and creates a 25:75 train/test split with caps of 1,000/3,000 examples.
- Apply caps without silently splitting groups or reassigning unused examples. Record actual counts and provenance.
- B100 means the selected training pool, not the entire original source dataset.
- Benchmark subsets are nested. Synthetic subsets share one frozen order within each pool.
- Replacement uses `N - floor(N * benchmark_fraction)` synthetic examples; addition uses `floor(N * synthetic_fraction)`. Preserve exact replacement totals when rounding.
- Require enough unique synthetic examples for the maximum configured replacement condition; do not repeat an undersized synthetic pool automatically.
- Named synthetic datasets are separate experiments, not an implicit mixture.
- Custom-split scores are not official benchmark-split scores. Do not tune on the final test set or claim that validation is generated automatically.

### Training and evaluation

- Baseline and joint execution must finish their training phase before its evaluation phase. Reuse the frozen backbone across selected LoRAs, unload without merging after each run, and create a new optimizer/scheduler for every adapter.
- Reset the training seed immediately before adapter initialization; token cache hits must not change initialization. Cache encoding and prompts only when data/template/length identities match.
- Reuse one vLLM engine for selected evaluations, with unique adapter IDs and bounded adapter slots. Keep the outer LoRA and inner batch tqdm bars descriptive; do not add a third request bar. Default inference chunks contain 64 prompts.
- All four joint conditions start from the original model with a fresh adapter. B100 is an evaluation reference only, identified by reference_id; joint jobs must not have parent_id. Match reference model revision, data identity, rank, alpha, and training seed.
- Use a fresh optimizer/scheduler for each joint run; restore optimizer state only when resuming that same run. Mix benchmark and synthetic rows deterministically, namespace their IDs, and retain source labels.
- Default training uses five epochs for both benchmark-only and joint runs, with per-device micro-batch 8 and gradient accumulation 2 (effective batch 16 on one GPU). Report unique-example ratios separately from repeated epoch exposure.
- Default benchmark supervision is a Direct prompt and a JSON final answer. Compute loss only on the final assistant response.
- Do not add reasoning targets, packing, or silent truncation as an incidental implementation change.
- Evaluate both Direct and CoT. CoT prompts request reasoning and a final-line JSON answer; native model thinking is disabled by default.
- Hold evaluation questions, prompts, decoding, and parser policy fixed across compared conditions.
- Separate extraction success, correctness, requested-format compliance, and length-limit outcomes.
- Parser extraction must not depend on the gold label or select an arbitrary last number. Preserve conservative ambiguity handling and exact numeric normalization.
- Version changes to scoring semantics must change `PARSER_VERSION` so cached reports and result cohorts remain distinguishable.
- Pair improvements only across matching seeds, recipes, and evaluation questions. Report available seed counts and the limits of question-bootstrap intervals.
- The benchmark-repeat control starts from the original model and matches the entire joint mixture to the nearest whole-example input-token budget. Do not describe it as exact compute matching.

## Reproducibility and storage

- Derive run identities from content and configuration, not machine-specific absolute paths. Keep model and published adapter revisions pinned to immutable commit SHAs during execution.
- Verify dataset and adapter integrity before use. Do not fabricate registry entries, completion markers, training results, or publication revisions.
- Preserve immutable artifacts and use atomic metadata writes and locks where appropriate.
- Store base LoRA training artifacts and reference evaluations in `outputs_base_lora/`; store joint LoRAs and repeat controls in `outputs/`. Both roots must be gitignored.
- Base training uses the current run folder as its source of truth: reuse verified completed artifacts, train absent runs, and retain interrupted recovery state. The registry is a derived index, never a reason to block training an absent run. Refresh it after verified local completion; archive conflicting old entries in its `.history.json` file and drop their stale remote associations before replacement. Do not modify remote publications or overwrite corrupt completed files.
- Keep recovery checkpoints for incomplete runs. Remove them only after final adapter completion and hash verification; preserve adapters, tokenizers, training metadata, and evaluation artifacts.
- Keep result discovery and upload discovery aware of both roots, preserve cross-root paired comparisons, and deduplicate identical copied reports. Output path changes must not alter run identities.
- Base adapters share `<namespace>/synthetic-joint-training-base-loras`; joint LoRAs use a separate configurable shared repository.
- Store checkpoints under model/benchmark/seed/condition/run-hash subfolders. Different recipes, including ranks, must remain distinguishable.
- Publish only selected adapter/tokenizer files and training metadata. Exclude original datasets, full model weights, optimizer checkpoints, and evaluation completions.
- Upload new checkpoint folders incrementally; skip identical registered artifacts and reject conflicting content for an existing run ID.
- Merge the shared remote registry without dropping unrelated entries. Retain optimistic concurrency checks and retries.
- Download the small registry first, then only the required checkpoint files at their recorded revisions. Reuse cached references instead of downloading the full repository.
- Keep dry runs network-free and GPU-free. Explicit publication commands control live Hub writes; do not publish as a side effect of tests or documentation edits.

## Validation

Run checks appropriate to the change from the repository root:

```bash
python -m pytest -q
python -m compileall -q src examples
for script in shells/*.sh; do bash -n "$script"; done
git diff --check
```

For plan or shell changes, use the offline demo:

```bash
python examples/make_demo_data.py
bash shells/prepare_dataset.sh \
  --local-benchmark-root data/demo/normalized \
  --synthetic-file data/demo/synthetic.jsonl --data-root data/demo-prepared
bash shells/run_base_lora.sh --data-root data/demo-prepared --dry-run
bash shells/run.sh --data-root data/demo-prepared --mode all --dry-run
```

Check that the default plans contain 6 base jobs and 24 joint jobs per model/benchmark/synthetic dataset, with matching rank/alpha/seed references. Use isolated temporary paths in automated tests.

Add focused regression tests when changing scoring, split integrity, reference selection, or remote registry behavior. Do not contact live Hub services or start GPU training in the default test suite.

When reporting validation, distinguish offline tests, mocked services, and dry runs from actual GPU execution or live publication. If the runtime is unavailable, state the concrete limitation instead of claiming that training or inference passed.
