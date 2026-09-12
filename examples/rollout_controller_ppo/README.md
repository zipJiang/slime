# Versioned rollout-controller PPO integration

This Slime example versions the synchronous direct-branch experiment integration.
The rollout-controller library owns tree construction and target estimation; this
example owns GPU allocation, native optimization, serving and weight synchronization.
`source-lock.json` pins the controller and upstream Slime baseline. Run
`scripts/snapshot_sources.py` after committing changes before using the SIF launcher.
The sibling `slime/.venv` and shared SIF provide the established reusable runtime.

## Batch overlap candidate

`--ppo-execution sync` retains synchronous native critic inference. The optional
`--ppo-execution overlap` uses five actor inference GPUs and one frozen critic
inference GPU, with the same four native training GPUs. Set `PPO_ROLLOUT_GPUS=5`
for overlap. All trees retain one actor and one critic version. A batch can be
at most one optimizer update old; native PPO uses its stored behavior logprobs.
The existing numerical mismatch limits remain enforced in both modes.

The next batch is collected while the current batch trains. Publication waits
for both tasks. Periodic saves, evaluations, the final batch, and a `STOP` file
in the run directory drain the pipeline before saving the exact question cursor.
A `STOP` arriving after prefetch begins takes effect after that batch is trained.
`--ppo-stop-after-round N` ends at zero-based collection round N without changing
the optimizer schedule; the saved attempt can be resumed normally.

Critic publication gathers only model tensors from the first Megatron DP replica,
converts the text backbone and scalar head, and verifies checksums and inference
against native final-context values. Initial verification includes long contexts;
`--ppo-critic-equivalence-contexts FILE` adds recorded real contexts. No optimizer
state is served. Each publication is logged; only the newest inference snapshot
is kept, while normal native paired checkpoints retain recovery state.

Before promotion, compare four joint updates in each mode from the same audited
warmup checkpoint and question cursor, with `--ppo-benchmark`, a shared
`--ppo-seed-namespace`, and identical token budgets. Keep six actor inference GPUs
for synchronous mode so both arms use ten GPUs total. Benchmark mode explicitly
defers boundary evaluation. Include pipeline fill/drain and publication costs;
report steady throughput over the two interior updates separately. Require at
least 20% higher interior accepted-token throughput and a positive net throughput
gain over the whole window, with comparable budgets and passing target, provenance, probability,
checkpoint and native/replica checks before prioritizing overlap. Otherwise
continue the synchronous algorithm-only arm. This is a throughput selection,
not evidence of an accuracy improvement.

The imported experiment notes below describe the original live run. Generated run
folders, logs and snapshots are excluded from Git.

# Direct-branch PPO, Qwen3.5-9B

Separate synchronous ablation of `deontic-ppo-9b`. This experiment starts from
base Qwen3.5-9B, uses six critic-only warm-up rounds with empirical suffix targets,
then 120 actor updates with direct-branch preparation. The actor uses recursive
posterior TD credit; the critic learns the direct-child mean before the current
node's prior is blended. Search allocation remains the original two-pass recipe.

The collector records `recipe_id=direct-branch-td-v1-suffix-warmup`, the selected
estimator, and critic target source. Resume rejects checkpoints from another
recipe. Training data replay is disabled; `preflight_replay.py` only creates
validation artifacts from historical trees.

## Sources and storage

- `snapshots/harness`: immutable source snapshot including DirectBranchTdEstimator.
- `snapshots/slime`: copy of the original experiment's tested pinned backend.
- `source-provenance.json`: origin and manifest hash.
- `runs/direct-branch-v1`: link to physical storage at
  `/weka/projects/bvandur1/zjiang31/deontic-ppo-direct-branch-9b/runs/direct-branch-v1`.
- `validation/{warmup,direct-branch}`: six-tree re-export, target replay, costs,
  source hashes, and no-mutation checks for each preparation mode.
- `logs`: Slurm/Ray launch output, training output, and audit watchers.

Before launch, the Python 3.13 harness integration check passes and the Slime
container contract tests pass. Historical re-export verifies both modes; this
does not establish improved training accuracy. Native critic readiness and exact
behavior log-probability checks run in the training driver.

## Allocation and launch

Use an isolated Ray cluster: gh202 has four H200 training GPUs; gh203, gh121,
and gh130 each supply two rollout GPUs. The actor and critic optimize sequentially
on gh202. There is no separate critic inference replica in this synchronous run.

Start Ray under each existing Slurm allocation, with `srun --overlap` and the
absolute script path. Preserve user allocation steps and other experiments.

```bash
bash scripts/ray_node.sh train
bash scripts/ray_node.sh rollout 172.16.204.2:6405 2
```

Run the first command on gh202 and the second on each rollout host. Ray uses port
6405, dashboard 8285, and `/tmp/deontic-direct-branch-ray`. The generic Slime node
size is two for rollout port grouping; native actor and critic arguments retain
four GPUs per training node.

On gh202, inside its Slurm allocation:

```bash
PPO_RUN_NAME=direct-branch-v1 bash scripts/run_ppo.sh
```

Defaults preserve the original six-question balanced batch, 140k output tokens
per pass per question, four concurrent expansions, single submission, 48+1 task
horizon, actor LR 1e-6, critic LR 5e-6, kappa=1, KL coefficient .01, and 120 actor
updates. Root-only heldout evaluation and checkpoints occur every 12 actor
updates, plus warm-up boundaries. Do not equate adaptive search success with
heldout accuracy.

Use a fresh attempt name for recovery, after paired checkpoint readback succeeds:

```bash
PPO_RESUME_RUN=/absolute/path/to/previous/run \
PPO_RUN_NAME=direct-branch-v2 bash scripts/run_resume_ppo.sh
```

The driver archives runtime Python sources and their hashes. Do not edit those
sources or the snapshots while a run is using them.

## Checks

From the rollout-controller checkout:

```bash
.venv/bin/python ../trl-train/experiments/deontic-ppo-direct-branch-9b/scripts/check_harness.py
```

Run tensor/transport tests through the reusable Slime SIF wrapper, using absolute
paths because the wrapper changes directory to its backend snapshot:

```bash
bash /absolute/experiment/scripts/sif.sh python -m unittest discover \
  -s /absolute/experiment/scripts -p 'test_*.py'
```

Use `watch_audits.py RUN`, `watch_checkpoints.py --run RUN`, and
`audit_evaluation.py --watch-run RUN` with the host Python 3.13 environment to
audit new artifacts. Checkpoint watchers invoke the SIF wrapper themselves.

The proposed async extension is documented separately in the rollout-controller
repository at `docs/research/bounded-async-tree-ppo.md` and is not enabled here.

## Critic inference consistency

The standalone critic pins `flash_attention_2` and scores the final context token
before an appended sentinel, matching native checkpoint positioning. Every
publication checks the full equivalence corpus, not just the two readiness
prompts. The absolute probability tolerance remains 0.005. Publication fails on
incorrect versions/counts, invalid probabilities, nondeterminism, or excessive
error, and records per-context differences.

`ray_node.sh` and `run_ppo.sh` use `with_torch_cudnn.py` inside the SIF. It sets both
`CUDNN_HOME` (which Transformer Engine checks first) and `LD_LIBRARY_PATH` to the
Python-bundled cuDNN. The image also has an older system cuDNN; mixing them made
loading import-order dependent. Start fresh Ray nodes with these launchers for a
future training run. Existing processes and standalone evaluation launchers are
unchanged.

`probe_critic_replica.py` provides read-only inference ablations against archived
native critic scores and checks tensor/context hashes before comparison. The
September 12 regression covers critic versions 6 and 78; future publications
must still pass their own live native comparison.
