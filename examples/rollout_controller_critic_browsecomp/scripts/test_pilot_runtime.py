"""Profile compatibility and numerical publication gates, without GPUs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from critic_equivalence import compare_scores
import pilot_runtime as runtime


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']


def payload(turns):
    return SimpleNamespace(messages=(),workspace=SimpleNamespace(snapshot=lambda:{}),
        turns_taken=turns,state=SimpleNamespace(calls_remaining=10,calls_max=10),
        done=False,truncated=False)


def test_critic_conditioning_uses_actual_long_horizon(monkeypatch):
    monkeypatch.setattr(runtime,'TASK_LIMIT',96)
    text=runtime.context(payload(81),[],Tokenizer())
    state=json.loads(text.split('\n',1)[1])
    assert state['remaining_task_steps']==15
    assert json.loads(runtime.context(payload(96),[],Tokenizer()).split('\n',1)[1])['remaining_task_steps']==0


def test_legacy_context_bytes_match_pretraining(monkeypatch):
    # Execute only the frozen serializer, without importing its path-mutating collector.
    import ast
    source=Path(__file__).with_name('collect.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='context')
    namespace={'json':json}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'collect.context','exec'),namespace)
    monkeypatch.setattr(runtime,'TASK_LIMIT',48)
    for turns in (0,12,48,49):
        assert runtime.context(payload(turns),[],Tokenizer())==namespace['context'](payload(turns),[],Tokenizer())


def result(scores, version='critic-test'):
    return dict(version=version,scores=scores)


def test_lenient_default_accepts_gpu_drift_and_retains_error():
    native,replica=result([.1,.5]),result([.10885315239429474,.5])
    report=compare_scores(native,replica,replica,version='critic-test',count=2)
    assert report['passed'] and report['tolerance']==.01
    assert report['max_abs_error']==pytest.approx(.00885315239429474)


def test_lenient_gate_still_rejects_drift_nondeterminism_and_bad_versions():
    native=result([.5])
    changed=result([.511])
    assert not compare_scores(native,changed,changed,version='critic-test',count=1)['passed']
    assert not compare_scores(native,native,result([.50001]),version='critic-test',count=1)['passed']
    with pytest.raises(ValueError,match='version'):
        compare_scores(native,result([.5],'other'),native,version='critic-test',count=1)
    with pytest.raises(ValueError,match='probability'):
        compare_scores(native,result([float('nan')]),native,version='critic-test',count=1)


def test_trace_profile_limits(monkeypatch):
    if not runtime.TRACE:
        pytest.skip('Run also with BROWSECOMP_PROFILE=trace96k')
    profile=runtime.verify_harness()
    assert profile['context_limit']>73728
    assert profile['task_limit']>80
    assert profile['actor_reply_limit']>8192
    assert profile['prompt_limit']==14336
