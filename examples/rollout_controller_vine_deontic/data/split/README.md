# Balanced deontic split for future runs

The new dataset uses all 676 available airline, SARA numeric, and SARA binary
cases, without selecting for successful model rollouts. It freezes a **50/50
train/test split: 338 cases and 248 case families per partition**. Original hard
labels are retained. The split is stored at
`data/splits/deontic-balanced-v1-20260912/`.

| Task | Train cases | Train hard | Test cases | Test hard |
|---|---:|---:|---:|---:|
| Airline | 150 | 41 | 150 | 39 |
| SARA numeric | 50 | 18 | 50 | 17 |
| SARA binary | 138 | 15 | 138 | 15 |
| Total | 338 | 74 | 338 | 71 |

Each partition contains 150 airline, 50 numeric, and 48 binary families. Binary
answers are balanced within each partition: 69 entailments and 69 contradictions.
The unique inventories retain all cases; balancing the training distribution is
done through sampling, so useful normal cases are not discarded.

## Why change the old pool?

The old imitation-derived pool contained 57 training families and 13 validation
families: 81.4%/18.6% of the selected 70-family cohort. Its training pool covered
only 11.5% of the 496 available families. The training-question file had 81 rows
because it repeated roots to balance domains; these were 57 unique root cases:

| Task | Old unique roots | Originally hard |
|---|---:|---:|
| Airline | 23 | 3 |
| SARA numeric | 27 | 12 |
| SARA binary | 7 | 3 |

The earlier preparation used successful demonstrations, and the RL pool inherited
their case selection. Original hard labels and model solvability are distinct:
the new split balances the former and eliminates the success-selection gate.
It does not assume every originally normal case is easy for Qwen3.5-9B.

## Assignment and leakage controls

The builder unions exact normalized fact scenarios, including across tasks, and
explicit SARA binary positive/negative siblings transitively. A family stays in
one partition. The current inventory has no cross-task families; the builder
fails explicitly if future data introduce one requiring joint stratification.

Previously trained families and the 13 historical validation families are locked
into training. Test quotas are chosen from the remaining families. Assignment
stratifies by task and whether a family includes a hard case, and additionally by
airline complexity. Because binary families can contain both hard and normal
cases, candidate assignments minimize deviations in case-level difficulty and
binary answer counts while preserving family quotas. Seed: `20260912`. No model
outcomes enter the assignment.

The new test cases were already evaluated in the expanded checkpoint sweep.
This is a fixed test set for future training, not an untouched benchmark. Once
training uses the new split, the old 600-case evaluation overlaps training and
must not be reported as held out. Use the new 338-case test list instead. For
frequent checkpoint selection, make a development split *within training* and
reserve this test for final comparisons.

## Training distribution and fixed compute

Sample a task uniformly; choose hard/normal with probability 1/2; choose a family
uniformly within that cell; then choose a case uniformly within that family's
cell. Each of the six task/difficulty cells gets 1/6 of the rollout roots.
Equal family mass prevents a binary scenario with many related questions from
dominating. A mixed hard/normal family participates in both cells, but every
member remains in the same train/test partition.

`train.jsonl` contains each training case once and records its exact
`metadata.sampling_weight`. Uniformly shuffling that file **does not implement
the balanced distribution**. The sampler generates a reproducible schedule with
one root from each cell per six draws, sampling within cells with replacement:

```bash
.venv/bin/python -m examples.deontic.balanced_split sample \
  --split-dir data/splits/deontic-balanced-v1-20260912 \
  --draws 6000 --seed 20260912 \
  --output /tmp/train-schedule.jsonl
```

A 6,000-draw example is included in the bundle. This is an offline question
schedule, not a request for 6,000 training rollouts. Set the actual number of
roots/updates or token budget to match the intended experiment. The larger pool
does not require a longer run: under a fixed budget, it gives more variety and
fewer repetitions per family. Harder cases may still consume more tokens, so use
an explicit token budget when comparing compute efficiency.

For future Slime integration, select the actual task using `metadata.case_key`,
and retain `metadata.family` for leakage grouping. The old experiment shim uses
`family` as a case selector and hard-codes its old split manifest. It must be
configured/adapted to this new allowlist before using these files. Repeated draws
are separate rollout roots; the old collector also rejects duplicate case keys
within a batch. The new bundle deliberately does not modify pinned experiments
or restart training.

## Evaluation and files

Evaluate every test case once per checkpoint. Report raw overall pass@1, all
three tasks, and each task's hard/normal accuracy. Also report the unweighted
mean of those six cell accuracies so easy binary cases cannot dominate the
headline balanced score. An optional family-balanced score is the sum of each
test outcome times its stored `sampling_weight`; this is different from the
six-cell case-accuracy mean. Bootstrap whole case families for uncertainty.

The bundle includes:

- `train.jsonl`, `test.jsonl`: unique question records with explicit case/family
  identity, original difficulty labels, and sampling weights.
- `train-case-keys.json`, `test-case-keys.json`: unambiguous allowlists.
- `corpus/<task>/{train,test}.jsonl` and `corpus/<task>/statutes/`: copied original
  case records and all referenced statutes; no dependency on a mutable corpus
  is needed to load this frozen data.
- `inventory.jsonl`: every case's partition, family, difficulty, and historical
  training/validation flags.
- `manifest.json`, `source-manifest.json`: assignment settings and source/output
  SHA-256 hashes.
- `build.py`: frozen copy of the builder; rebuilding verifies the original
  source files before reading them and refuses to overwrite an existing split.
- `audit.py`, `audit.json`: source/content checks, historical family-rule
  equivalence, disjointness, balanced schedule checks, and a byte-identical
  rebuild in a temporary directory.

Builder:
[`examples/deontic/balanced_split.py`](../../../examples/deontic/balanced_split.py).
It uses only the Python standard library. Rebuild into a new directory with:

```bash
.venv/bin/python -m examples.deontic.balanced_split build \
  --source-manifest data/evaluation/deontic-expanded-20260912/manifest.json \
  --output /tmp/deontic-balanced-rebuild \
  --test-fraction 0.5 --seed 20260912
```

Four focused tests cover transitive family leakage, locked training families,
sampling mass, deterministic schedules, and infeasible quotas. The real-data
audit verifies all 676 cases and the frozen artifact hashes.
