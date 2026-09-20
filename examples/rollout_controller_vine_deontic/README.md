# DeonticBench VinePPO baseline

This experiment implements the compute-comparable VinePPO baseline for the
balanced DeonticBench split. The next continuation resumes the last accepted
checkpoint after round 26 of
`vine-both-null-g5-c12-sep19-replacement-633715`, changes the retained root
group from five to four, and keeps one independent value rollout per live
state. It is deliberately queued after the LocBench GRPO run. The authoritative
queue contract is `operations/queued-after-loc-grpo-sep20.json`.

## Data and sampling

The frozen split has 338 train and 338 test cases with no family overlap. Each
six-question training update contains one root from every domain/difficulty
cell. Within a cell the schedule samples a family uniformly and then a case
uniformly within that family, with replacement. Therefore 120 updates mean 720
root selections, not two exhaustive dataset passes. The checked-in split
manifest and corpus are verified by `scripts/balanced_data.py` before
collection.

## Algorithm contract

For each selected question, VinePPO retains four root trajectories. At every
live state it generates one independent continuation solely to estimate that
state's binary success probability. The auxiliary continuation is discarded;
only the retained root trajectory contributes actor tokens. Step credit is
`V(s[t+1]) - V(s[t])`.

Both the actor and compactor use `NullWorkspace`, matching the corrected
comparison requested for this baseline. Generation uses a 48-task-turn horizon
plus one forced submission, a 14,336-token compaction trigger, 6,144-token task
reply limit, 4,096-token fold reply limit, and 32K model context. Training uses
TP2/DP2, LR 1e-6, KL coefficient .01, clip .2, one optimizer update per six
questions, and 120 total planned updates.

The recipe identifier is
`vine-ppo-balanced-k1-base-both-null-g4-v5-20260920`. A resume from the accepted
group-5 checkpoint must pass `--resume-group-size-from 5`; the transition is
recorded in the new run's provenance. The rejected round-27 successor must
never be used as a resume source.

## Launch and validation

Use `operations/supervise_vine_group4_sep20.py` with four trainer GPUs on the
head reservation and the remaining assigned GPUs for rollout. The first
continuation is exactly one synchronous update from round 26:

```bash
python operations/supervise_vine_group4_sep20.py \
  --name <operation-name> --run-name <run-name> \
  --resume-run /weka/projects/bvandur1/zjiang31/deontic-vine-9b/runs/vine-both-null-g5-c12-sep19-replacement-633715 \
  --group-size 4 --value-rollouts-per-state 1 \
  --resume-group-size-from 5 --search-concurrency 12 \
  --execution sync --num-rollout 28 \
  --assign <trainer-job>=<allocation-width>:<four-devices> \
  --assign <rollout-job>=<allocation-width>:<devices> \
  --head <trainer-job>
```

Do not start the asynchronous continuation until the new checkpoint passes the
full tensor readback and the synchronous batch passes the exact stored-behavior
log-probability audit. The continuation then resumes that checkpoint with
`--execution overlap --num-rollout 120`.

Run the focused source tests with:

```bash
pytest --import-mode=importlib \
  scripts/test_balanced_data.py scripts/test_batches.py \
  scripts/test_checkpoint_restore_memory.py scripts/test_resume_vine.py \
  scripts/test_targets.py
```

The native backend and harness snapshots used by live runs remain pinned under
`snapshots/`; every run copies its effective runtime sources and recipe into the
run directory for audit.
