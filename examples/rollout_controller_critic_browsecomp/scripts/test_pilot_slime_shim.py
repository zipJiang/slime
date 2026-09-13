from types import SimpleNamespace

import pytest

from pilot_slime_shim import expected_batch, selected_questions


def schedule():
    return dict(schema='browsecomp-zero-warmup-pilot-schedule-v1',updates=2,batch_size=2,
                batches=[['a','b'],['c','d']])


def test_exact_round_batch_is_selected():
    assert expected_batch(schedule(),0,2)==['a','b']
    assert expected_batch(schedule(),1,2)==['c','d']


@pytest.mark.parametrize('rollout_id',[-1,2])
def test_round_outside_pilot_is_rejected(rollout_id):
    with pytest.raises(ValueError,match='outside'):
        expected_batch(schedule(),rollout_id,2)


def test_slime_group_identity_is_exact():
    groups=[[SimpleNamespace(metadata={'query_id':'a'})],
            [SimpleNamespace(metadata={'query_id':'b'})]]
    assert selected_questions(groups)==['a','b']


def test_ambiguous_data_group_is_rejected():
    with pytest.raises(ValueError,match='one query'):
        selected_questions([[SimpleNamespace(metadata={'query_id':'a'}),
                             SimpleNamespace(metadata={'query_id':'a'})]])
