from types import SimpleNamespace
import pytest

from fresh_actor import FreshStartActor
from slime.backends.megatron_utils.actor import MegatronTrainRayActor


def test_fresh_hf_actor_reports_cursor_zero(monkeypatch):
    monkeypatch.setattr(MegatronTrainRayActor,'init',lambda self,args,role,with_ref,with_opd_teacher: 1)
    actor=object.__new__(FreshStartActor)
    args=SimpleNamespace(finetune=True,no_load_optim=True,no_load_rng=True)
    assert actor.init(args,'actor')==0


def test_wrapper_rejects_the_critic_role():
    actor=object.__new__(FreshStartActor)
    args=SimpleNamespace(finetune=True,no_load_optim=True,no_load_rng=True)
    with pytest.raises(ValueError,match='policy role'):
        actor.init(args,'critic')


def test_wrapper_reports_fresh_optimizer_and_scheduler():
    class Optimizer:
        state={1:{'step':0},2:{'step':0.0}}
    actor=object.__new__(FreshStartActor)
    actor.optimizer=Optimizer()
    actor.opt_param_scheduler=SimpleNamespace(num_steps=0)
    assert actor.audit_optimizer_start()['fresh']


def test_wrapper_detects_restored_optimizer_history():
    class Optimizer:
        state={1:{'step':2}}
    actor=object.__new__(FreshStartActor)
    actor.optimizer=Optimizer()
    actor.opt_param_scheduler=SimpleNamespace(num_steps=12)
    report=actor.audit_optimizer_start()
    assert not report['fresh'] and report['steps']==[2.0]
