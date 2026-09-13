"""Actor wrapper with an explicit fresh model-only initialization cursor."""
from slime.backends.megatron_utils.actor import MegatronTrainRayActor

from warmstart import initialization_cursor


class FreshStartActor(MegatronTrainRayActor):
    def audit_optimizer_start(self):
        optimizers=getattr(self.optimizer,'chained_optimizers',[self.optimizer])
        steps=[]
        for wrapper in optimizers:
            optimizer=getattr(wrapper,'optimizer',wrapper)
            for state in optimizer.state.values():
                if 'step' in state: steps.append(float(state['step']))
        scheduler_steps=float(self.opt_param_scheduler.num_steps)
        return dict(steps=sorted(set(steps)),scheduler_steps=scheduler_steps,
                    fresh=all(step==0 for step in steps) and scheduler_steps==0)

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        if role != 'actor':
            raise ValueError('FreshStartActor can only initialize the policy role')
        cursor=super().init(args,role,with_ref,with_opd_teacher)
        return initialization_cursor(args,cursor)
