"""LocBench outcome contract: rejected model compactions terminate with zero recall."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
HARNESS = EXPERIMENT / 'snapshots/harness-vine'
sys.path.insert(0, str(HARNESS))
MODEL = '/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
CONTEXT_LIMIT = 98304
TASK_LIMIT = 80
REPLY_LIMIT = 16384


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return dict(profile='locbench-focused-coding-v2', compaction_failure='terminal_zero', model=MODEL,
        tools=['list', 'grep', 'read', 'submit'], reward='file_recall@5',
        task_limit=TASK_LIMIT, horizon='global task turns plus folds; one final submission outside budget',
        server_context_limit=CONTEXT_LIMIT, actor_reply_limit=REPLY_LIMIT,
        fold_reply_limit=REPLY_LIMIT, prompt_limit=32768, call_budget=20,
        summary_target_tokens=2048, actor_temperature=.6, actor_top_p=.95,
        fold_temperature=.7, fold_top_p=1., top_k=20, repeat_limit=0,
        context_schema='expected-file-recall-v1',
        harness_sha256=digest(HARNESS/'source-manifest.json'))


def verify_harness():
    manifest=json.loads((HARNESS/'source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest(HARNESS/relative)!=expected:
            raise ValueError(f'Frozen harness differs: {relative}')


def context(payload, tools, tokenizer):
    """Only observable checkpoint state; future return and gold stay private."""
    state=dict(messages=list(payload.messages), tools=tools,
        workspace=dict(payload.workspace.snapshot()),
        remaining_steps=max(0,TASK_LIMIT-payload.turns_taken-payload.folds),
        calls_remaining=payload.state.calls_remaining, calls_max=payload.state.calls_max,
        last_request=payload.state.last_request,
        consecutive_requests=payload.state.consecutive_requests,
        done=payload.done, truncated=payload.truncated)
    content='Predict the expected final file recall@5 from continuing this localization checkpoint, between 0 and 1.\n'
    content+=json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return tokenizer.apply_chat_template([dict(role='user',content=content)],
        tokenize=False,add_generation_prompt=True)


def make_world(case, repo, policy):
    from examples.locbench.env import RunConfig, build_world, LocBenchRunner
    from step_controller.harness.compaction.base import IncompleteCompactionError
    class HorizonRunner(LocBenchRunner):
        async def compact(self, rs):
            try:
                return await super().compact(rs)
            except IncompleteCompactionError as exc:
                if not exc.turns:
                    raise
                return compaction_terminal(rs, exc.turns, exc.reasons, self.policy.version)

        async def advance(self, rs, *, sampling_params=None):
            if rs.done or rs.truncated:
                return rs
            if rs.turns_taken+rs.folds>=TASK_LIMIT:
                return await self.finish(rs,sampling_params=sampling_params)
            return await super().advance(rs,sampling_params=sampling_params)
    world=build_world(case,repo,RunConfig(policy=policy,compact=True,max_steps=TASK_LIMIT,
        call_budget=20,max_prompt_tokens=32768,fold_reply_tokens=REPLY_LIMIT,
        focused_guidance=True,repeat_limit=0))
    # Build the experiment's global-horizon runner from the same local world.
    original=world.runner
    runner=HorizonRunner(repeat_limit=0,policy=policy,env=original._env,
        system_prompt=original._system_prompt,compactor=original._compactor,
        sampling_params=original._sampling_params,max_steps=TASK_LIMIT,
        finish_prompt=original._finish_prompt,tools=original._tools)
    return runner,world.workspace,world.prompt,original._tools


def compaction_terminal(rs, turns, reasons, version):
    """Rejected notes never replace context; genuine model tokens stay trainable."""
    from step_controller.harness import Turn, StepResult
    if not reasons:
        raise ValueError('A typed model compaction failure needs explicit reasons')
    state=replace(rs.state,locations=(),submitted=False)
    marker=Turn(prefix=(),logprobs={version:()},transition=StepResult(next_state=state,done=True,
        reward_outcome=0.,messages=({'role':'user','content':'Compaction failure: '+', '.join(reasons)},)))
    return replace(rs,state=state,turns=rs.turns+tuple(turns)+(marker,))
