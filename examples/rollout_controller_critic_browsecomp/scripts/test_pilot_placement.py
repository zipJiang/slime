import pytest
from pilot_placement import rollout_bundle_order, training_bundle_order
from pilot_topology import allocation_plan, required_gpus


def test_rollout_engines_group_by_host_then_gpu():
    physical=[('train',0),('train',1),('a',1),('b',0),('a',0)]
    assert rollout_bundle_order(physical,2,3)==[4,2,3]


def test_training_prefix_is_not_returned_as_rollout():
    physical=[('train',0),('train',1),('rollout',0)]
    assert rollout_bundle_order(physical,2,1)==[2]


def test_training_pairs_stay_local_across_two_hosts():
    physical=[('b',1),('a',1),('b',0),('a',0),('inference',0)]
    assert training_bundle_order(physical,4)==[3,1,2,0]
    with pytest.raises(ValueError,match='one host'):
        training_bundle_order([('a',0),('a',1),('a',2),('b',0)],4)
    with pytest.raises(ValueError,match='repeat'):
        training_bundle_order([('a',0),('a',0),('b',0),('b',1)],4)


@pytest.mark.parametrize('train,replica,expected', [
    ([1],None,dict(train=4,inference=3,aux=2)),
    ([1,4],None,dict(train=2,train_worker=2,inference=3,aux=2)),
    ([1],5,dict(train=4,inference=2,replica=1,aux=2)),
    ([1,4],5,dict(train=2,train_worker=2,inference=2,replica=1,aux=2)),
])
def test_nine_gpu_pilot_allocation_layouts(train,replica,expected):
    jobs,required=allocation_plan(train,2,3,replica)
    assert required==expected==required_gpus(jobs)
    assert sum(required.values())==9


def test_invalid_pilot_layouts_are_rejected():
    with pytest.raises(ValueError,match='distinct'):
        allocation_plan([1,2],2,3)
    with pytest.raises(ValueError,match='one four-GPU'):
        allocation_plan([1,4,5],2,3)
    with pytest.raises(ValueError,match='incomplete'):
        required_gpus({'train','aux'})


def test_seven_gpu_pilot_uses_one_local_training_pair():
    jobs,required=allocation_plan([1],2,3,4,train_gpus=2)
    assert required==dict(train=2,inference=2,aux=2,replica=1)
    assert sum(required.values())==7
    assert training_bundle_order([('train',1),('train',0)],2)==[1,0]
    with pytest.raises(ValueError,match='one host'):
        allocation_plan([1,5],2,3,4,train_gpus=2)
