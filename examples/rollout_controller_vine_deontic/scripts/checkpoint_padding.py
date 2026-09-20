"""Initialize only Megatron's explicitly marked, unused checkpoint padding."""
from functools import wraps


def zero_padding(state):
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensor

    if isinstance(state, dict):
        marker = state.get('padding')
        if isinstance(marker, LocalNonpersistentObject) and marker.unwrap() is True:
            count = 0
            for value in state.values():
                if isinstance(value, ShardedTensor):
                    if value.data is None:
                        raise ValueError('Save-time optimizer padding has no storage')
                    value.data.zero_()
                    count += value.data.numel()
            return count
        return sum(zero_padding(value) for value in state.values())
    if isinstance(state, (list, tuple)):
        return sum(zero_padding(value) for value in state)
    return 0


def install():
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original = DistributedOptimizer.sharded_param_state_dp_reshardable
    if getattr(original, '_browsecomp_zero_padding', False):
        return

    @wraps(original)
    def initialized_padding(self, model_sharded_state_dict, is_loading=False, metadata=None):
        state = original(self, model_sharded_state_dict, is_loading=is_loading, metadata=metadata)
        if not is_loading:
            zero_padding(state)
        return state

    initialized_padding._browsecomp_zero_padding = True
    DistributedOptimizer.sharded_param_state_dp_reshardable = initialized_padding
