# Loc-Bench V1 experiment

Code localization from an issue and the repository at `base_commit`, using exactly
`list`, `grep`, `read`, and `submit`. Evaluation trajectories and training trees use
the same environment and file-recall reward.

## Dataset and metric contract

Source: [czlll/Loc-Bench_V1, default/test](https://huggingface.co/datasets/czlll/Loc-Bench_V1/viewer/default/test),
pinned to revision `c44cf3b74e07ca642cec841b471a9939907c12a7` and a checked SHA-256.
There are **560 cases across 165 repositories**. This setting preserves the upstream
`test` cohort; it does not invent a train/dev split. `--instances` and `--limit`
select a cohort in dataset order, recorded in the run manifest.

Each episode ends on its first valid call to:

```json
{"locations": ["src/config.py::Config.__init__", "src/helpers.py"]}
```

The default limit is **10 entries**, shared by all methods. Entries are ordered
relative paths or `path::Qualified.name`. No prose answer is accepted. An empty
array is an explicit abstention. Malformed submissions receive feedback and can be
retried within the episode budget. Valid nonexistent paths/functions are scored as
predictions, rather than rejected using private labels.

The [LocAgent paper, §5.1](https://arxiv.org/html/2503.09089v2) defines Acc@k as
finding **all** relevant locations in the first k predictions. Following the supplied
experiment contract:

| Granularity | Ranking | Gold | Reported metrics |
|---|---|---|---|
| File | Distinct paths in first-occurrence order; qualified entries contribute their path | Files changed by `patch` | Acc@1/3/5 and recall@1/3/5 |
| Function | Distinct qualified entries; remove bare paths and exact `added_functions` matches before ranking | `edit_functions` only | Acc@5/10 and recall@5/10 |

**Training reward = file recall@5.** Wrong qualified functions consume function ranks.
Constructors retain `.__init__`; class names are not aliases for constructors. A
bare path still contributes at file level even though it contributes no function rank.
Duplicate entries never inflate recall. New functions are excluded only from function
scoring, not from their containing file's relevance.

Gold file paths use the base side for edits, deletions, and renames; newly added files
use their new path. `test_patch` is excluded. New files need not exist at the base
commit and may still be submitted. All patch files count, including non-Python files.
There are 24 cases with more than five gold files (maximum 15); file Acc@5 is zero
on those cases, and recall@5 remains partial credit. The denominator is never capped
at k. This intentionally follows the supplied all-target definition rather than the
gold-vector truncation in the upstream `evaluation/eval_metric.py` implementation.

Aggregate metrics are means over the selected cases. Missing/malformed submissions
and failed episodes remain in the denominator. A case with no edited functions has
no function score; its file score remains eligible. Every case in this pinned dataset
has edited-function labels.

## Evidence tools

| Tool | Fixed behavior |
|---|---|
| `list(path)` | No argument: top level. Explicit subtree: depth 2. At most 200 entries, with total at that depth, truncation flag, and narrowing hint. |
| `grep(pattern, path=None, max_hits=50)` | POSIX extended regex over base-commit text. `path:line: text` hits; at most 50. Counts all matching lines and asks for a narrower pattern/path when truncated. |
| `read(path, start=1, lines=60)` | 1-based line window, at most 60 lines, total file length, and next starting line. Requests above 60 are rejected; use `next_start` to continue. No sections or symbol lookup. |
| `submit(locations)` | Ranked array of up to 10 bare or qualified locations; ends the episode. |

The repository is fetched into a bare Git cache and addressed by its full commit
SHA. Tools do not read a mutable checkout or `HEAD`. There is no manifest in the
prompt, graph API, shell, Python execution, editing, Git history, `find`, or storage
tool. Compaction uses a tool-free `NullWorkspace`; the task's original issue is
retained. Patches, hints, and function labels stay in the scorer, outside model
prompts and tool responses.

Paths are literal, repository-relative POSIX paths. Traversal and `.git` access are
rejected; symlinks and submodules are listed but never followed. Binary files cannot
be read as source. Reads cap file size at 32 MiB, and very long output lines are
explicitly clipped at 2,000 characters. Git commands have a 30-second timeout
(initial fetch: 300 seconds). No repository code is executed or installed.

**Page-size probe:** row 11's vLLM base commit has 722 files, 462 Python files. A
deterministic sample of 64 Python files had line-count quartiles 63/141/293, p90 388,
and maximum 1,390. Of those, 35 exceeded the previous 120-line page size. The current
limit is **60 lines**; `probe` reports lengths and does not auto-tune the interface.

## Prepare and run

From the repository root, with Python 3.12+ and Git:

```bash
uv sync --extra locbench --extra vllm --extra chat
uv run python -m examples.locbench download
```

The downloaded file is
`data/locbench/source/test-c44cf3b74e07ca642cec841b471a9939907c12a7.parquet`.
For a small reproducible pilot using row 11:

```bash
LOCBENCH_DATA=data/locbench/source/test-c44cf3b74e07ca642cec841b471a9939907c12a7.parquet
uv run python -m examples.locbench prepare --data "$LOCBENCH_DATA" \
  --instances vllm-project__vllm-5473
uv run python -m examples.locbench probe --data "$LOCBENCH_DATA" \
  --instances vllm-project__vllm-5473
```

Run against an existing **fixed-checkpoint vLLM endpoint** using its matching local
model/tokenizer path. The endpoint must support the library's token-native completion
protocol. The CLI runs a startup check before collecting:

```bash
uv run python -m examples.locbench run --data "$LOCBENCH_DATA" \
  --instances vllm-project__vllm-5473 \
  --model /path/to/model --profile qwen_xml --served-model locbench-model \
  --vllm-url http://HOST:PORT/v1 \
  --output data/locbench/eval-no-compaction

uv run python -m examples.locbench run --data "$LOCBENCH_DATA" \
  --instances vllm-project__vllm-5473 \
  --model /path/to/model --profile qwen_xml --served-model locbench-model \
  --vllm-url http://HOST:PORT/v1 --compact \
  --output data/locbench/eval-compaction
```

Omit `--instances` to run all 560, or use `--limit N` for the first N cases. Both
methods default to 80 runner advances (including fold-only advances), plus one
finalization attempt, 8,192 generated tokens per response, actor temperature 0,
and concurrency 4. Compaction triggers at 32,768 prompt tokens or 20 tool calls
since the last fold; it has 8,192 generation tokens, a 2,048-token summary target,
and the library's default fold temperature of 0.7.
The per-round call budget triggers compaction only; without compaction the overall
step budget controls termination. The final budget-forced generation accepts only
`submit`. Choose server context capacity with room for prompt growth, tool output,
and generation; `--max-prompt-tokens` is the compaction trigger, not a server limit.
Compare methods on the same IDs, model, server capacity, task budgets, and tool limits.

`--focused-guidance` opts into localization-specific stopping guidance and
evidence-only compaction. The fold transcript is explicitly delimited as history;
the summary records candidate locations, observed evidence, completed searches and
read ranges, and existing uncertainties. It adds no new research checklist. This
option is shared by evaluation and training through `RunConfig.focused_guidance`.
It changes prompts only: the four tools, scoring, and budgets stay fixed.

`--repeat-limit 6` optionally ends investigation after six consecutive identical
repository requests and makes one ordinary final-submission attempt. Zero (the
default) disables this additional budget. Request identity normalizes valid defaults;
changing the pattern, path, or read window resets the counter. The counter is in
immutable branch-local task state and survives compaction. Tool results are preserved.
The final attempt executes no extra research calls; a missing valid submission scores
zero. This bounds a degenerate loop but does not guarantee a correct answer.

### Paired pilots across inference replicas

`examples.locbench.pilot` accepts a JSON server manifest with `model` (local
tokenizer path) and `urls` (a list of vLLM `/v1` endpoints). Each must serve the
fixed model as `locbench-qwen35-9b`. It assigns each question to the same endpoint
in both arms, alternates arm admission order, and logs every completed generation
including sampling settings, completion text, token counts, stop reason, and
latency. Its generous reply default is 16,384 tokens for actor and fold alike.
Temperature-zero inference is still subject to GPU and batching differences.
Use `--temperature`, `--top-p`, and `--top-k` to record an explicit actor decoding
configuration. Fold overrides retain temperature 0.7 and top-p 1.0; top-k inherits
the policy setting.

```bash
uv run python -m examples.locbench.pilot \
  --data data/locbench/pilot-20260915/discovery.jsonl \
  --servers data/locbench/pilot-20260915/operations/attempt2/server-manifest.json \
  --output data/locbench/new-paired-pilot --focused-guidance
uv run python -m examples.locbench.report data/locbench/new-paired-pilot \
  --output data/locbench/new-paired-report.json
```

The [September 15 pilot](../../docs/research/locbench-pilot-20260915.md) completed
96 trajectories over 20 questions. On its eight validation questions, focused
compaction with the coding preset scored 3/8 all-file Acc@5 at 21.4 mean task turns
and 4,768 generated tokens per question. This was cheaper than focused greedy
compaction at the same all-file success, with lower partial recall and function
accuracy. It is a small-sample operating recommendation, not a full-benchmark claim.
To reproduce that configuration on a new paired batch with active endpoints:

```bash
uv run python -m examples.locbench.pilot \
  --data /path/to/questions.jsonl --servers /path/to/server-manifest.json \
  --output data/locbench/new-coding-preset-pilot \
  --focused-guidance --temperature 0.6 --top-p 0.95 --top-k 20 \
  --max-tokens 16384 --max-steps 80 --call-budget 20 \
  --max-prompt-tokens 32768
```

The report requires completed cohorts and checks predictions against native
submissions. Errors remain in accuracy denominators; length and native-trace
statistics expose their smaller denominator when traces are missing. Repetition
analysis distinguishes exact successful calls, calls repeated from before a fold,
and previously observed lines in overlapping reads. A repeat is evidence to inspect,
not proof of wasted work. Malformed submits and tool-shaped fold replies are counted
separately. Reports re-analyze all native traces using one analysis version.

Each run uses a new output directory containing a cohort/configuration manifest,
atomic per-case result JSON, native `RolloutState` checkpoints for completed episodes
(`traces/*.pkl.gz`), ordered `predictions.jsonl`, and `summary.json`. Results include
turns, tool counts, folds, prompt/completion tokens, forced endings, token exactness,
latency, and errors. Per-case files persist as jobs finish; the aggregate files are
written after the cohort finishes. An interrupted run can retain completed cases,
but this initial CLI does not automatically resume an interrupted cohort.

Offline scoring accepts rows with `instance_id` and `locations`:

```bash
uv run python -m examples.locbench score --data "$LOCBENCH_DATA" \
  --instances vllm-project__vllm-5473 \
  --predictions data/locbench/eval-compaction/predictions.jsonl
```

## Training integration

`env.build_world(case, repository, RunConfig(policy=...))` is shared by evaluation
and `bench.search_one`. The latter accepts the library's `RolloutConfig`, advantage
estimator, and optional value model, and returns a `PreparedBatch`. For
`RefinedTdEstimator`, supply a value model for nonterminal checkpoints; GRPO over
independent root rollouts can use `GrpoEstimator` without a value model. Search and
sample preparation retain the same local runtime and behavior version.

Pass the batch through the existing `step_controller.to_samples` exporter. Its
advantage cutoff and actor/critic handling apply without a separate LocBench
implementation. Task and compaction generations remain native trainable turns;
repository tool responses remain masked conditioning. The environment changes only
the task and terminal reward, not allocation, value backup, or sample normalization.

## Validation

```bash
uv run pytest tests/examples/test_locbench.py -q
uv run ruff check examples/locbench tests/examples/test_locbench.py
uv run mypy examples/locbench tests/examples/test_locbench.py --follow-imports=silent
```

Tests cover ranking and exclusions, multi-file reward, invalid/forced submissions,
patch additions/deletions/renames, base-commit isolation, literal paths, symlink
isolation, bounded outputs/counts, concurrent tool calls, compaction, native trace
persistence, the training preparation path, download integrity, and CLI outputs.
These are harness checks; no model success rate is claimed from scripted answers.
