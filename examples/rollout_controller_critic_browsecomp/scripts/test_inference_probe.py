import pytest
from inference_probe import probe_indices


def test_covers_every_question_root_and_longest_fold():
    rows=[dict(group_index=q,turn=t) for q,t in [('a',0),('a',2),('a',4),('b',0),('b',2),('c',0)]]
    # A long root must not displace the representative folded context.
    assert probe_indices(rows,[1000,20,30,1000,40,1000])==[0,2,3,4,5]


def test_rejects_missing_roots_and_misaligned_lengths():
    with pytest.raises(ValueError,match='root'):
        probe_indices([dict(group_index='a',turn=2)],[10])
    with pytest.raises(ValueError):
        probe_indices([dict(group_index='a',turn=0)],[])
    with pytest.raises(ValueError):
        probe_indices([],[])
