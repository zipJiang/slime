# LocBench critic and PPO experiment

Train a reusable file-recall critic from fresh base Qwen3.5-9B continuations, then
start native PPO with exact behavior logprobs and one-batch collection overlap.
This experiment is isolated from the live BrowseComp run. Large artifacts live on
project storage through `data`, `runs` and `logs` symlinks; `snapshots` pins the
LocBench harness, native Slime and shared training utilities.

The active research log is in the rollout-controller checkout at
`docs/research/locbench-ppo-20260915.md`. Runtime commands, job IDs, audits and failure
evidence are under `operations/`. These machine-specific artifacts are not source.

- `collect_warmup.py` and `runtime.py` are the immutable original Monte Carlo
  collection sources. Do not edit them while a collection manifest references them.
- `runtime_v2.py` treats typed incomplete model compactions as zero-recall terminals;
  future PPO retains exact rejected generations. Transport errors still fail.
- `reconcile_warmup.py` derives a separate critic-only corpus, preserving successful
  native traces and exact prior checkpoints for recorded incomplete compactions.
  Historical missing rejected tokens are never synthesized as actor supervision.
- `audit_warmup_v2.py` independently checks every derived outcome and native prefix.
- `train_critic.py` uses held-out calibration to select a checkpoint and verifies
  tensor readback, model-only optimizer reset and native/portable predictions.
- `collect_ppo.py`, `audit_ppo.py` and `reexport_ppo.py` retain native search evidence,
  independently replay TD preparation and apply whole-edge library cutoffs.
- `efficiency.py` and `ppo_batches.py` analyze retained signal and preserve original
  question/edge loss normalization, including sparse or entirely filtered groups.
- `train_ppo.py` supports isolated, reverted optimizer benchmarks and sustained PPO
  with explicit actor/critic counters and drained paired checkpoint recovery.

`run_ppo.sh` requires `LOC_PPO_OUTPUT` and a dedicated `RAY_ADDRESS`; the selected
critic is supplied with `LOC_CRITIC_CANDIDATE`. Set `LOC_PPO_BENCHMARK_ONLY=1` for an
isolated timing attempt. A reviewed plan in `LOC_EFFICIENCY_PLAN` fixes budget,
cutoff and GPU counts for training. Never interpret benchmark optimizer steps as
committed PPO updates. `operations/ppo_supervisor.py` checks device ownership before
launch and requests a drained checkpoint 90 minutes before the earliest lease ends.

The PPO driver and launch path still require real GPU validation. Passing CPU or
argument checks alone is not evidence that a run is trained or reusable.
