"""Versioned environment and horizon for fresh pilot rollouts.

The original critic collection keeps its original context() function and snapshot.
The TRACE profile reuses that JSON schema with an accurate, longer task horizon.
"""
import json
import os
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
PROFILE = os.environ.get('BROWSECOMP_PROFILE', 'legacy')
if PROFILE not in ('legacy', 'trace96k'):
    raise ValueError(f'Unknown BrowseComp profile: {PROFILE}')
TRACE = PROFILE == 'trace96k'
HARNESS = EXPERIMENT/'snapshots'/('harness-trace-v1' if TRACE else 'harness')
TASK_LIMIT = 96 if TRACE else 48
CONTEXT_LIMIT = 98304 if TRACE else 32768
SERVER_CONTEXT_LIMIT = 98304 if TRACE else 65536
ACTOR_REPLY_LIMIT = 16384 if TRACE else 6144
FOLD_REPLY_LIMIT = 4096
PROMPT_LIMIT = 14336
TOP_K = 10 if TRACE else 5


def environment_contract():
    return dict(profile=PROFILE, toolset='trace-docids-v1' if TRACE else 'legacy',
        task_limit=TASK_LIMIT, context_limit=CONTEXT_LIMIT,
        server_context_limit=SERVER_CONTEXT_LIMIT,
        actor_reply_limit=ACTOR_REPLY_LIMIT, fold_reply_limit=FOLD_REPLY_LIMIT,
        prompt_limit=PROMPT_LIMIT, top_k_documents=TOP_K,
        submit='submit', single_tool_per_turn=TRACE)


def context(payload, tools, tokenizer):
    state = dict(messages=list(payload.messages), tools=tools,
        workspace=dict(payload.workspace.snapshot()),
        remaining_task_steps=max(0, TASK_LIMIT-payload.turns_taken),
        calls_remaining=payload.state.calls_remaining, calls_max=payload.state.calls_max,
        done=payload.done, truncated=payload.truncated)
    content = 'Predict the probability that a continuation from this checkpoint succeeds.\n'
    content += json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return tokenizer.apply_chat_template([dict(role='user', content=content)],
        tokenize=False, add_generation_prompt=True)


def verify_harness():
    import hashlib
    manifest = json.loads((HARNESS/'source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = HARNESS/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen harness source differs: {relative}')
    if TRACE and manifest.get('environment_profile') != 'trace96k':
        raise ValueError('TRACE requires a separately prepared harness snapshot')
    return environment_contract()
