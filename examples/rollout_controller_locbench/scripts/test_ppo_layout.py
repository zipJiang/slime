import pytest
from ppo_layout import validate_layout


def test_split_training_keeps_replica_and_rollout_capacity_separate():
    assignments={'1':[0,1],'2':[0,1],'3':[0,1,2,3],'4':[1]}
    result=validate_layout(assignments,['1','2'],'4',2)
    assert result['total_gpus']==9
    assert result['rollout_gpus']==4


@pytest.mark.parametrize('train,replica,count',[
    (['1','1'],'4',2), (['1'],'1',2), (['1'],'4',4),
    (['1','2','3'],'4',2), (['missing'],'4',2),
])
def test_invalid_training_ownership_rejected(train,replica,count):
    with pytest.raises(ValueError):
        validate_layout({'1':[0,1],'2':[0,1],'3':[0,1],'4':[1]},train,replica,count)


@pytest.mark.parametrize('devices',[[0,0],[-1,0],[True,1],[]])
def test_invalid_selected_devices_rejected(devices):
    with pytest.raises(ValueError):
        validate_layout({'1':devices,'2':[0,1],'3':[1]},['1'],'3',2)


def test_training_host_spare_devices_become_rollout_capacity():
    result=validate_layout({'1':[0,1,2,3],'2':[1]},['1'],'2',2)
    assert result['rollout_gpus']==2


def test_expiring_rollout_hosts_can_be_removed_without_changing_training():
    from ppo_layout import retain_launchable_rollouts
    assignments={'1':[0,1],'2':[0,1],'3':[0,1],'4':[1],'5':[0,1]}
    leases=[dict(job=j,remaining_seconds=100 if j=='3' else 20000) for j in assignments]
    kept,excluded=retain_launchable_rollouts(assignments,['1','2'],'4',leases,10800)
    assert excluded=={'3':100}
    assert validate_layout(kept,['1','2'],'4',2)['rollout_gpus']==2
    leases[0]['remaining_seconds']=100
    with pytest.raises(ValueError,match='Training and critic'):
        retain_launchable_rollouts(assignments,['1','2'],'4',leases,10800)


def test_missing_lease_inventory_cannot_silently_drop_devices():
    from ppo_layout import retain_launchable_rollouts
    with pytest.raises(ValueError,match='inventory'):
        retain_launchable_rollouts({'1':[0,1],'2':[1]},['1'],'2',[],10800)
