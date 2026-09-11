# Versioned rollout-controller PPO integration

This Slime example versions the synchronous direct-branch experiment integration.
The rollout-controller library owns tree construction and target estimation; this
example owns GPU allocation, native optimization, serving and weight synchronization.
`source-lock.json` pins the controller and upstream Slime baseline. Run
`scripts/snapshot_sources.py` after committing changes before using the SIF launcher.
The sibling `slime/.venv` and shared SIF provide the established reusable runtime.

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
