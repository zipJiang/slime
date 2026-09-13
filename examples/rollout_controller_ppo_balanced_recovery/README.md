# Balanced PPO checkpoint recovery

This recovery preserves interrupted `balanced-base-v4` training: base Qwen3.5-9B,
the balanced split, ten critic-only warmup rounds, the original constant learning
rate schedule, and 120 configured actor updates. The user chose to retain ten
warmup rounds after the six-round attempt encountered a saved scheduler mismatch.

The original run stopped after critic update five: one standalone HF prediction
differed from native inference by 0.0050627589, above its 0.005 check. A read-only
probe reproduced the discrepancy with the saved tensors and exact contexts.
Changing kernels changed the error; no universal exact match was established.
The user authorized a modest numerical allowance. This recovery explicitly sets
`--ppo-critic-equivalence-tolerance 0.01`; the function default remains 0.005.
Every publication still checks all 17 contexts, exact snapshot checksums and
versions, finite probabilities, and deterministic repeated standalone inference.
Reports retain every error, maximum error, mean error, and the selected tolerance.
This is approximate numerical agreement, not a claim of bitwise equivalence.

The first recovery segment completed 24 actor and 34 critic updates, with paired
checkpoint 33, optimizer state, question cursor, and export readbacks verified.
`balanced-base-warmup10-recovery-v4` attempted a round-34 resume with two training
GPUs on each of gh106/gh108. Placement succeeded, but native actor optimizer load
exceeded an 80 GB H100's memory before any update. That attempt is preserved.
`balanced-base-warmup10-recovery-v5` uses the fallback: four H200 training GPUs
on gh202, three actor inference GPUs across gh101/gh129, and one standalone
critic GPU on gh129. World size four and TP=2/DP=2 are retained, with no repeated
warmup. gh106/gh108 are available for BrowseComp work. The two-host placement
path still needs a successful memory-feasible optimizer restore before production use.

The supervisor accepts one four-GPU or two two-GPU `--train-job` arguments,
one or more `--rollout-job` allocations (two GPUs each), a `--critic-job` from
that inference pool, and an explicit `--resume-run`. It derives IP addresses and
expiry times from the running allocations, and requests a drained save 90 minutes
before the first expiry by default. CPU audits run within the supervisor allocation.
Both native optimizers must retain their saved step counters and scheduler samples
on every rank before resumed collection. The original v3 launcher files are
preserved in `operations/balanced-topology-original/`; its Python runtime snapshot
is also retained within the original run.

Recovery sequence:

1. Restore native checkpoint iteration zero: one critic update, zero actor
   updates, optimizer/RNG state, and the saved question cursor.
2. Replay saved batches 1–4 using the same native loss and packing. Check exact
   question order; preserve original contracts and source checksums.
3. Compare each reconstructed critic against its archived native predictions on
   all 17 fixed contexts, with an explicit replay tolerance of 0.01. The default
   remains 1e-5 for strict comparisons.
4. Save a full paired checkpoint and cursor after batch four, before collecting
   fresh data, then finish the remaining five warmup rounds.
5. Evaluate the base actor at the warmup boundary and start actor updates at
   collection round ten. The first supervised segment ends after 24 actor updates.

Warmup targets are observed continuation outcomes; stored critic priors are not
used as warmup training targets. Target/horizon audits run again on copied data.
The original experiment remains unchanged at
`../rollout_controller_ppo_balanced/runs/balanced-base-v4`.

Current attempt: `balanced-base-warmup10-recovery-v5`, supervisor **402684**.
At September 13, 12:27 EDT, all four ranks passed native optimizer restoration
(actor step 24, critic step 34, scheduler samples 144/204). The supervisor passed
its independence audit and fresh round-34 collection is active. The default stop
request is September 13 at 17:27 EDT, before gh202 expires at 18:57 EDT.
Inspect `operations/` and `runs/` for subsequent updates and checkpoints.
The failed six-warmup attempt is preserved for diagnostics and made no updates.

Validation: 15 resume, data-replay, and numerical-comparison tests passed; seven
pipeline/recipe tests passed in the native CUDA image. The four saved batches
previously passed full target/horizon replay audits. Live checks must still verify
optimizer reconstruction, new full checkpoint readback, and fresh collection.

The first ten-warmup restart loaded both roles and restored all 17 native critic
predictions exactly. Its first replay update had identical loss, batch size, LR,
training sources, and all 24 training-data files compared with the original run.
The gradient norm differed by 0.002016%; subsequent predictions differed by at
most 0.00194347, triggering the original strict replay check. The current attempt
uses the user-authorized 0.01 allowance for this check as well. Evidence is in
`runs/balanced-base-warmup10-recovery-v2/restart-numerics-diagnostic.json` and
`initial-native-restore-audit.json`. Approximate restart is not bitwise trajectory
reconstruction. Invalid probabilities, versions, context counts, or data remain
hard failures independent of the numerical allowance.

Historical verification at September 12, 22:37 EDT: v3 loaded both checkpoints and
reproduced all 17 initial native predictions exactly. All nine launchers passed
the independent-supervisor cgroup audit. The first replayed batch passed its
whole-batch target audit and optimizer replay check: maximum error 0.0018538833,
mean error 0.0006199917, tolerance 0.01, critic optimizer update count two. Three
saved batches still need replay, then five fresh warmup rounds. The new paired
checkpoint after replay is not yet verified. Current run tracking:
https://wandb.ai/zipjiang/deontic-compaction-ppo/runs/p8kv76cp


## Migration to the idle NVLs, September 13

Supervisor **403770** (`balanced-base-warmup10-recovery-v6`) waits for v5 supervisor
402684 to finish its requested drained stop. It verifies that the paired native
checkpoints cover the paused actor/critic counters, both full tensor readbacks
passed, and the question cursor exists before automatically releasing GO.
It retains TP=2/DP=2, optimizer history, learning rates, and asynchronous overlap.

The replacement trains on two H100 NVLs each on gh130/384963 and gh119/384964.
A real four-rank restore probe passed for both roles; peak reserved memory was
82.80 GiB for the actor and 75.22 GiB for the critic. These 94 GB cards fit the
restore that failed on 80 GB H100s. Evidence: `runs/nvl-restore-probe-v1/complete.json`.

Four actor rollout engines use gh101 and gh129. Critic inference uses only physical
GPU 1 on gh108/401540; BrowseComp keeps GPU 0. A separate-cluster CUDA probe verified
that isolation. `--critic-gpu` requires an explicit valid index for a separate
critic allocation; its Ray worker uses a distinct dashboard agent port.

`--resume-after-job 402684` gates this cutover on successful source completion and
checkpoint audits. `--release-idle-job 360795` returns gh202 only after the
replacement's first completed actor update and native restore audit, and only
when the old allocation has no active steps. Fourteen focused allocation, port,
and migration-boundary tests pass. Inspect live reports before claiming migration
or allocation return is complete.
