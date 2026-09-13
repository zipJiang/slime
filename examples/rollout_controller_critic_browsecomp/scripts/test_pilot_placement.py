from pilot_placement import rollout_bundle_order


def test_rollout_engines_group_by_host_then_gpu():
    physical=[('train',0),('train',1),('a',1),('b',0),('a',0)]
    assert rollout_bundle_order(physical,2,3)==[4,2,3]


def test_training_prefix_is_not_returned_as_rollout():
    physical=[('train',0),('train',1),('rollout',0)]
    assert rollout_bundle_order(physical,2,1)==[2]
