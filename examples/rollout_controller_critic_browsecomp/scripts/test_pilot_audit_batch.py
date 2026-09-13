import hashlib
import json

import pytest

from pilot_audit_batch import audit_judge
from semantic_judge import JUDGE_MODEL, JUDGE_VERSION


def record(question='Where?', references=('Paris',), submitted='Paris',
           response='EQUIVALENT', correct=True):
    return dict(version=JUDGE_VERSION,model=JUDGE_MODEL,
        question_sha256=hashlib.sha256(question.encode()).hexdigest(),
        references_sha256=hashlib.sha256(json.dumps(tuple(references)).encode()).hexdigest(),
        submitted=submitted,response=response,finish_reason='stop',correct=correct,attempt=1)


def test_judge_replay_accepts_exact_versioned_evidence():
    audit_judge([record()],question='Where?',references=('Paris',),
        terminal_pairs=[('Paris',1.)])


@pytest.mark.parametrize('change',[
    dict(model='other'),dict(version='other'),dict(response='DIFFERENT'),
    dict(finish_reason='length'),dict(attempt=4),dict(submitted='')])
def test_judge_replay_rejects_tampered_evidence(change):
    evidence=record();evidence.update(change)
    with pytest.raises(ValueError):
        audit_judge([evidence],question='Where?',references=('Paris',),
            terminal_pairs=[('Paris',1.)])


def test_judge_replay_rejects_unmatched_terminal():
    with pytest.raises(ValueError,match='Terminal outcomes'):
        audit_judge([record()],question='Where?',references=('Paris',),
            terminal_pairs=[('London',0.)])
