# BrowseComp-Plus base critic pretraining

## TRACE-style fresh pilot profile (September 13)

`run_pilot.sh` now defaults to `BROWSECOMP_PROFILE=trace96k`: `browser.search`,
line-numbered `browser.open`, literal `browser.find`, and native `submit`.
Open/find take explicit stable docids, so navigation has no hidden cursor shared
between branches. One tool call is accepted per task turn. Search defaults to ten
results (maximum twenty); reads retain the 6,000-character cap and find returns at
most twenty bounded matches with continuation offsets.

The actor/server/trainer profile uses a 98,304-token context ceiling, 96 task turns
including final submission, and a 16,384-token actor reply ceiling. The task horizon
applies to the complete root-to-terminal path, including resumed branches. Early
compaction remains at 14,336 prompt tokens or ten tool calls, and fold replies are
limited to 4,096 tokens. Context exhaustion reduces the reply allowance or fails
explicitly; it never truncates conditioning. Vocabulary log-probability computation
uses 1,024-token chunks. A configured 96K ceiling does not establish that every
96K training microbatch fits the assigned GPUs; peak-memory validation is separate.

Fresh pilot critic publication now uses maximum absolute error **0.01**, matching
PPO recovery. Version, context count, finite probability, and deterministic repeated
inference checks remain mandatory; all numerical errors remain recorded. This
changes publication tolerance, not semantic answer grading or training LR.
The launcher defaults both actor and critic LR to 1e-6.

Prepare the immutable overlay before launching:

```sh
/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python \
  examples/rollout_controller_critic_browsecomp/scripts/prepare_trace_harness.py \
  --source /weka/scratch/jhu/bvandur1/zjiang31/rollout-controller
```

This creates `snapshots/harness-trace-v1` from the existing frozen harness, replacing
only the environment and adding its browser tools. It refuses to overwrite an
existing snapshot. Every new collection and replay verifies its source hashes.
The profile propagates through the native container and Ray workers. The collection
contract records the profile and serializer sources; the critic receives the actual
96-turn remaining horizon using the existing JSON schema. The original pretraining
serializer and collection snapshot remain byte-for-byte intact. Promotion records
the environment profile it actually exercised; old pilot results do not authorize
the new profile. `BROWSECOMP_PROFILE=legacy` selects the previous environment for a
fresh legacy attempt; old artifacts require their archived runtime sources.

The earlier v4 pilot stopped at its second publication with max error 0.00885315
against 0.005. Its failure report is retained; changing the new default does not
retroactively mark that pilot successful.

This isolated experiment collects fresh Qwen3.5-9B episodes with online compaction,
then trains a native Megatron scalar critic for reuse with a fresh base actor.
No actor optimizer is created. The existing deontic PPO experiment is separate.

The frozen question split contains 498 training, 83 development, and 249 final-test
questions. The initial collection uses the first 128 training and 32 development
IDs in the frozen order, with four independent episodes per question (640 total).
Final-test questions are excluded. Private benchmark text and generated artifacts
stay under ignored `data/` and `runs/` directories.

`scripts/prepare_split.py` reproduces the split from the private case file. It
checks unique IDs and normalized questions and reports word-set Jaccard overlap
of at least 0.8. This is a lexical check, not an exhaustive semantic duplicate audit.

## Execution

September 13 recovery: the original collector stopped after 577 completed traces
when one fold's final-summary prompt exceeded 32K. The saved traces passed exact
snapshot/context readback (6,412 contexts; maximum 5,007 tokens). Normal fold
interaction retains its 32K threshold; final summaries can now use a 64K overflow
window without truncating conditioning. Collection and pilot servers support this
window; critic conditioning remains limited to 32K. Episode failures now drain
other in-flight work before failing collection, and never become negative labels.

Run `base-v2` records this source transition in its collection manifest. Original
sources, infrastructure, manifest, and the retained-summary hash inventory are in
`collection/recovery/overflow-v1/`; prior supervisor logs and terminal state are in
`recovery/overflow-v1/`. New traces carry the resumed manifest hash. Both readback
and the training boundary verify provenance across the transition. Recovery
collection supervisor **401517** completed successfully at 11:01 EDT. All 640
traces passed exact readback: 7,195 contexts, 6,688 folds, 153 successful episodes,
maximum critic context 5,007 tokens. The initial training completed 16 updates and
full checkpoint readback, but held-out MSE 0.2186 was worse than the train-fitted
constant's 0.1096. Its strict 1e-5 reload check also stopped finalization on one
0.00125 prediction difference. Original evidence remains under `training/`.

Read-only packing and train/eval comparisons agreed. An isolated real update
reduced same-batch MSE from 0.2179 to 0.0900, and sleep/wake preserved all updated
predictions exactly. This motivated a lower-rate refinement, not a declaration
that the critic is ready.

Refinement supervisor **402798** completed on September 13 at 13:40 EDT. It used
gh106/gh108 (four H100s) and gh203/gh205 (four H200s), separate from PPO, to make
another 16-update pass at LR **1e-6** from critic checkpoint 15 with fresh optimizer
state. Outputs are under `runs/base-v2/training-refine-lr1e6-v1`; sibling operations
end in `-operations`. Final held-out MSE is **0.08692385**, versus the constant's
**0.10955276**. The final improvement bootstrap interval narrowly crosses zero,
and initial-state predictions remain overconfident; the gains are in fold states.

Full checkpoint readback passes, model-only reload reproduces all 1,428 predictions
exactly with fresh optimizer/cursor state on eight ranks, and portable inference
matches native values within **0.00409136** on 64 held-out contexts. The generated
`warmstart-candidate.json` passes independent reconstruction from its evidence.
Pilot **402969** passed host preflight but stopped before model initialization:
the native container could not see the host's `/projects` retriever alias.
The launcher now uses its canonical `/weka/projects` path. The actual native
container reproduces the full host preflight exactly after this fix. Replacement
supervisor **403190** initialized both models successfully on two ranks. Its
initial critic export matches every pretrained artifact hash, and initial
native/portable prediction error is at most 0.00183064. Collection then rejected
the seven-GPU infrastructure because its validator still assumed four trainer
GPUs. That validator now checks the declared supported trainer layout; 15 focused
tests and the actual collector check inside the native container pass.
Pilot **403238** subsequently reached sample preparation but failed because the
shared SFT fixture wrapper labeled live actor traces with the default `policy`
version. The pilot now binds the actual Slime policy directly, saves complete
search evidence before preparation, and drains other questions on one failure.
The two-pass search/save/reload/preparation test passes for actor-0000 and
actor-0001, including an actual bounded compaction turn, both terminal outcomes,
and exact generated-token coverage. It also passes inside the actual training
container; 16 focused tests pass.

Current supervisor **403526** runs `browsecomp-zero-warmup-refine-lr1e6-v4` on the
same seven GPUs. At September 13, 14:47 EDT, fresh generation is verified active.
Initial critic publication passes with maximum error **0.00183064** and exact
pretrained weight-file hashes. `startup-audit.json` records recipe, source,
question-disjointness, artifact, and live-process ownership checks. Both actual
joint updates, training-time peak memory, and final promotion remain unverified.

- `CRITIC_TRAIN_JOBS` supplies two or four distinct two-GPU allocations. The head
  IP and training host count are derived from those allocations.
- `CRITIC_TRAIN_OUTPUT` and `CRITIC_TRAIN_OPERATIONS` select fresh output paths.
  `CRITIC_TRAIN_EXTRA_ARGS` is a JSON array of native driver CLI strings.
- The launcher explicitly sets reload tolerance 0.005 and retains every error.
  The driver's default remains 1e-5 when invoked directly.
- Slurm CPU batch jobs own all service launchers. GPU steps use `--mem=0` to avoid
  inheriting the CPU supervisor's smaller memory request. Each service has its
  own process group and file log.
- Touch `runs/base-v2/STOP` to stop collection/training at the supervisor's next
  poll. The GPU allocations themselves are not canceled.

Current launchers are `operations/collect.sbatch` and `operations/train.sbatch`.
Both require `CRITIC_EXPERIMENT_ROOT` to be the absolute experiment directory.
The active launch also sets `CRITIC_RUN_NAME=base-v2`; collection skips the
already exercised pilot with `CRITIC_SKIP_PILOT=1`.
The training supervisor waits for complete collection and service teardown.

## Data and target contract

`scripts/collect.py` freezes the actor revision, split, sampling protocol, and
collection source hashes. Each episode uses 48 ordinary task turns plus at most
one forced submission, a 32K context window, 14,336-token compaction trigger,
6,144-token actor replies, and 4,096-token fold replies. Tool output is bounded.
The actor also performs its own compactions.

Only the root and nonterminal fold checkpoints become critic examples. Their
conditioning strings include the visible messages, tool schemas, workspace,
remaining task turns, and remaining tool budget. They exclude the answer key,
terminal outcome, and future continuation. A future BrowseComp PPO adapter must
use the exact `collect.context` serialization, including the same tokenizer
chat template and final-context value position.

The judge sees the question, reference answers, and submitted answer. Greedy
non-thinking decoding is constrained to `EQUIVALENT|DIFFERENT`. Empty submissions
are failures; incomplete or malformed judge outputs and retrieval failures raise
errors and cannot become negative labels. The judge is a separately versioned
training choice, not an official BrowseComp-Plus benchmark score.

Each checkpoint receives its fresh continuation's terminal outcome. Identical
prefixes within a question are merged into an empirical mean; no prefixes are
merged across questions. Roots therefore usually have four observations, while
interior prefixes often have one. Both successes and failures are retained.
Loss averages checkpoints within each question and then averages questions.
Every actor/fold request uses the pinned namespace plus lane, question, sample,
and call index to derive its vLLM seed. `scripts/audit_sampling.py` verifies the
four planned continuation streams remain distinct within every question.

## Training and artifacts

The first pass uses 16 optimizer updates, eight questions per update, Adam at
5e-6, unclipped probability MSE, BF16, TP=2, DP=4, and the existing native critic
loss/packing implementation. Additional epochs require an explicit experiment
decision after validation. This does not add extra critic updates to a PPO batch.

The driver evaluates the same held-out development prefixes before and after
training, reports question-weighted MSE/MAE, ten-bin calibration gaps, and
root/fold strata, and compares against a constant fitted only on training data.
Calibration uses the same equal-question, equal-checkpoint-within-question measure
as optimization. A paired question bootstrap reports uncertainty in improvement
over the training-fitted constant.

Validated refinement files under `runs/base-v2/training-refine-lr1e6-v1/`
(the original failed run remains under `runs/base-v2/training/`):

- `dataset-inventory.json`: every aggregated context hash, question, target,
  observation count, and root/fold position used by training or validation.
- `sampling-audit.json`: all planned deterministic seed-stream lineages and the
  within-question independence check; its hash is part of the warmstart evidence.
- `storage-preflight.json`: training-start filesystem capacity; at least 300 GiB
  must remain before native model allocation begins.
- `native/`: native model and optimizer checkpoint.
- `native/iter_0000015-readback.json`: independent full read of every saved
  model and optimizer tensor, including finite-value and optimizer-step checks.
- `reload-audit.json`: weight-only reload preserves predictions and starts with
  fresh optimizer/scheduler state.
- `native-validated.json`: native training and reload have completed.
- `inference/`: portable text backbone and scalar value head.
- `inference-audit.json`: portable/native comparison at the root and longest
  saved fold context of every development question, with tolerance 0.005.
  Retains exact context hashes, token lengths, both predictions, and each error.
- `complete.json`: quality metrics and usable artifact paths. The inference path
  is null if portability validation fails; the native artifact remains distinct.
- `warmstart-candidate.json`: fail-closed handoff for a future PPO launcher. It
  records the native checkpoint and required model-only load flags, exact context
  source-file and function hashes, evidence hashes, offline quality gates, and explicitly leaves
  long-run zero-warmup authorization false until an on-policy pilot passes.

Future fresh PPO initialization loads the native critic with `finetune=True`,
`no_load_optim=True`, and `no_load_rng=True`, while independently starting the actor
and rollout cursor from base/zero. This is not paired PPO checkpoint resumption.
`scripts/ppo_initialization.py` implements this distinct role split and requires
`num_critic_only_steps=0`; the ordinary paired-resume path must not be used for it.
It hashes the pilot adapter's live `context()` function and requires an exact
match with the function used for pretraining.
The future pilot must pass `fresh_actor.FreshStartActor` to Slime's
`create_actor_model`, so fresh HF actor ranks explicitly report cursor zero too.
The driver exercises this load path and checks actual optimizer/scheduler state.
The critic wrapper also normalizes model-only native loading to rollout zero:
Megatron finetune reports iteration zero, which unmodified Slime converts to
rollout one. Real paired-resume cursors remain unchanged. Training and reload
require every configured rank to report fresh cursor zero.

Pretraining does not by itself prove warmup can be removed. That decision still
requires the focused audit and a zero-warmup PPO pilot with the intended rollout
and node-selection protocol.
`scripts/pilot_gate.py` requires at least two audited joint updates, exact
candidate/context lineage, fresh actor and critic initialization, complete batch
and on-policy checks, native/portable critic publication equivalence, and full
paired-checkpoint readback before emitting a separate long-run manifest.

## Zero-warmup gate pilot

`scripts/pilot_data.py` freezes two disjoint batches of six training questions.
They do not overlap the 160 critic-collection questions or the final-test split.
`scripts/pilot_collect.py` executes two 140K-token search passes per question with
the same bounded task environment, direct-branch TD preparation, exact behavior
log probabilities, the pretrained critic prior, and the frozen semantic judge.
`scripts/pilot_audit_batch.py` reconstructs every actor and critic record from the
saved native tree before either optimizer may consume it.

The pilot uses seven or nine GPUs: two or four shared/offloaded native actor-and-critic
trainer GPUs, two SGLang rollout GPUs, one frozen portable critic GPU, and one
retriever plus one judge GPU. Four-rank training can use one host or two hosts;
each TP pair stays on one host. The portable critic may share the inference
allocation or use a separate allocation. This supports three to five distinct
hosts, including five two-GPU allocations (nine GPUs used, one unused). Artifacts default to
`/weka/projects/bvandur1/zjiang31/browsecomp-critic-ppo/runs`, linked from this
experiment's `runs/` directory. The supervisor requires at least 300 GiB free.

The current candidate is
`runs/base-v2/training-refine-lr1e6-v1/warmstart-candidate.json`. For another pilot,
use a fresh run name and a CPU batch job with already-running single-host allocations:

```bash
export CRITIC_EXPERIMENT_ROOT=/weka/scratch/jhu/bvandur1/zjiang31/slime-ppo-worktree/examples/rollout_controller_critic_browsecomp
export PILOT_CRITIC_LR=1e-6
sbatch operations/pilot.sbatch \
  --run-name browsecomp-zero-warmup-pilot-v1 \
  --candidate "$CRITIC_EXPERIMENT_ROOT/runs/base-v2/training-refine-lr1e6-v1/warmstart-candidate.json" \
  --train-job TRAIN_JOB --inference-job INFERENCE_JOB --aux-job AUX_JOB \
  --deadline-unix UNIX_TIMESTAMP
```

For two-GPU allocations, repeat `--train-job` for the second training host and
add `--replica-job` for the portable critic host:

```bash
sbatch operations/pilot.sbatch \
  --run-name browsecomp-zero-warmup-pilot-v1 \
  --candidate "$CRITIC_EXPERIMENT_ROOT/runs/base-v2/training-refine-lr1e6-v1/warmstart-candidate.json" \
  --train-job TRAIN_A --train-job TRAIN_B \
  --inference-job ROLLOUT_JOB --replica-job CRITIC_JOB --aux-job AUX_JOB \
  --deadline-unix UNIX_TIMESTAMP
```

For seven total GPUs, use a single two-GPU H200 training allocation (TP=2/DP=1):

```bash
sbatch operations/pilot.sbatch \
  --run-name browsecomp-zero-warmup-pilot-v1 \
  --candidate /absolute/path/to/training/warmstart-candidate.json \
  --train-gpus 2 --train-job TRAIN_H200_JOB \
  --inference-job ROLLOUT_JOB --replica-job CRITIC_JOB --aux-job AUX_JOB \
  --deadline-unix UNIX_TIMESTAMP
```

Both layouts retain six questions and one update per batch. Thirty focused tests
pass for topology, preflight, and promotion, and the native scheduler preserves
all sample identities and question weights at DP=1 and DP=2. Two-GPU model initialization is verified; training-time peak memory remains
to be tested by the live pilot.

`PILOT_CRITIC_LR` overrides the pilot critic learning rate (default 1e-6).
The pilot following `training-refine-lr1e6-v1` uses 1e-6, matching refinement.

The supervisor validates allocation size and separation, performs the shared
CPU preflight, starts and probes both auxiliary services and the five- or seven-GPU Ray
cluster, runs exactly two sequential joint updates, and tears down only the child
process groups it created. The driver checks fresh actor and critic cursors and
optimizers, publishes and compares the critic before collection and after each
update, audits current versus behavior log probabilities, saves both native
checkpoints, and reads every stored tensor. It emits `long-run-warmstart.json`
only when all mechanical gates pass and the pilot observes both successful and
failed terminal branches.

## Initial checks

Use the controller's Python for `scripts/preflight.py`,
`scripts/audit_collection.py`, and the data tests. Use `scripts/sif.sh` for the
native loss tests. `scripts/run_train.sh --critic-preflight-only` validates the
real training arguments and DP packing; Megatron's argument validation requires
visibility of a GPU, but this mode does not create or train a model.

Collection source files are pinned once the manifest is written. Changing those
files while collection runs makes a subsequent resume fail. Audit them read-only
first; fixes require an intentional source/data transition.

## Fresh TRACE critic warmup (September 13, evening)

The user requested critic warmup before joint training on the new environment.
Pilot **405170** was stopped before any joint update; its artifacts and explicit
user-requested stop record are preserved. Independent deontic PPO **404358** keeps
its nine GPUs.

Fresh collection **405234**, `runs/trace96k-critic-warmup-v1`, uses all seven other
GPUs: four H200s on gh203/gh205 and gh106 GPU0 generate base Qwen3.5-9B episodes;
gh106 GPU1 judges outcomes; gh108 GPU0 serves retrieval. Gh108 GPU1 belongs to PPO.
Five independently served base-model replicas each accept eight concurrent traces.
The frozen question inventory remains 128 train / 32 development, four independent
samples each, with no final-test access. This is fresh collection, not relabeling
or replaying the old 48-turn traces. The profile is TRACE search/open/find,
98,304 context tokens, 96 task turns, and up to 16,384 actor response tokens.
Compaction remains enabled at 14,336 prompt tokens or ten calls, with 4,096-token
fold responses. Complete snapshots, exact critic strings, retrieval observations,
judge outcomes, hashes, and remaining 96-turn budgets are saved and read back.

The dependent critic-only phase reuses six GPUs (three local TP2 pairs, DP3) and
uses gh108 GPU0 for validation. The native DP3 preflight accepts the unchanged
8-question optimizer batch, without discarding or duplicating training records.
The H100 pair on gh106 has NV6 connectivity; the H200 pairs report NODE links.
Tensor parallelism stays within each host. Validation evaluates immutable exports
while the trainer proceeds through at most the next four-update block.

Warmup loads `base-v2/training-refine-lr1e6-v1/native`, checkpoint 15, with fresh
optimizer/RNG/cursor state. That retains the prior critic's useful refinement.
Only the critic updates. LR is **1e-6**, probability MSE is unclipped during
warmup, questions retain equal total loss, and roots receive **25%** of each
question's loss while its fold checkpoints divide the other 75%. Questions with
only one stratum retain total weight one. The root weighting is an explicit
hypothesis motivated by the earlier roughly 5% root loss mass and overconfident
roots, not an established improvement.

There are at most two passes / 32 updates. Evaluate every four updates, stop after
two validated boundaries without improvement (up to one additional block already
in flight), and retain the earliest balanced validation MSE improvement of at
least 1e-4. The starting critic participates in selection. Standard equal-question
MSE/MAE, calibration, and separate root/fold metrics remain reported alongside the
new balanced selection measure. Native checkpoints are saved at every validation
boundary. The chosen checkpoint undergoes full tensor readback, model-only reload
with fresh cursors and optimizer, and full held-out prediction comparison. Portable
publication additionally checks each dev question's root and longest fold, exact
repeated scores, versions, finite probabilities, and maximum error **0.01**.

A candidate must improve aggregate and balanced validation MSE over the starting
critic and train-fitted constant, and improve root MSE over the starting critic.
If the starting critic wins, the run records that outcome and produces no new
candidate. The supervisor never starts joint actor training automatically.
A new candidate's serializer is `scripts/pilot_runtime.py`; a later pilot must set
`PILOT_CONTEXT_SOURCE` to this file (the default remains `collect.py` for older
candidates). The next pilot still starts its actor from base.

Entry points are `operations/trace_warmup_supervisor.py` for collection and
`operations/trace_train_supervisor.py` for its successful-completion dependency.
Collection and training operations have distinct Slurm ownership and output
folders. Only explicitly assigned device slots and child process groups are used.
The 96K configuration and native packing preflight do not establish worst-case
96K backward memory fit; actual compaction checkpoint lengths are recorded.
