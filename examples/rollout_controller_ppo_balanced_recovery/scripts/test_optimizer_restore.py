from copy import deepcopy
import pytest
from optimizer_restore import validate_restore


def evidence():
    resume = dict(actor_updates=24,critic_updates=34)
    reports = {role:[dict(steps=[count],states=12,scheduler_samples=count*6,
        no_load_optim=False,no_load_rng=False,finetune=False) for _ in range(4)]
        for role,count in [('actor',24),('critic',34)]}
    return resume,reports


def test_four_restored_ranks():
    resume,reports = evidence()
    assert validate_restore(resume,reports,6)['passed']


@pytest.mark.parametrize('field,value', [('steps',[0]),('states',0),('scheduler_samples',0),
                                       ('no_load_optim',True),('no_load_rng',True),('finetune',True)])
def test_one_rank_cannot_silently_reset(field,value):
    resume,reports = evidence()
    reports['actor'][2][field] = value
    with pytest.raises(ValueError):
        validate_restore(resume,reports,6)
