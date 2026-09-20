"""Allocate Megatron's disposable restore placeholders on CPU.

The pinned DistributedOptimizer builds full GPU Adam placeholders before
Transformer Engine allocates its real state, transiently duplicating the state.
Only those empty placeholders move to CPU. TE still creates real optimizer
state on each parameter's CUDA device; native checkpoint loading overwrites it.
The exact source guard fails closed if the pinned Megatron implementation changes.
"""
import hashlib
import inspect
import re
import textwrap


def install():
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    original = DistributedOptimizer.load_state_dict
    if getattr(original, '_vine_cpu_restore_placeholders', False):
        return
    source = textwrap.dedent(inspect.getsource(original))
    pattern = r'(init_shard = lambda dtype=torch.float32: torch.empty\(\s*\(numel,\), dtype=dtype, device=)torch.cuda.current_device\(\)'
    replaced, count = re.subn(pattern, lambda match: match[1] + '"cpu"', source)
    if count != 1 or original.__closure__:
        raise RuntimeError('Pinned Megatron restore placeholder implementation changed')
    namespace = {}
    exec(compile(replaced, __file__, 'exec'), original.__globals__, namespace)
    patched = namespace['load_state_dict']
    patched._vine_cpu_restore_placeholders = True
    patched._source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    DistributedOptimizer.load_state_dict = patched


def restore_aliased_metadata(optimizer, state_dict):
    """Return True only when input state aliases every current state tensor."""
    import torch
    if (not optimizer.state or optimizer._optimizer_load_state_dict_pre_hooks
            or optimizer._optimizer_load_state_dict_post_hooks):
        return False
    groups = optimizer.param_groups
    saved = state_dict['param_groups']
    if len(groups) != len(saved):
        return False
    pairs = []
    for live, old in zip(groups, saved):
        if len(live['params']) != len(old['params']):
            return False
        pairs.extend(zip(live['params'], old['params']))
    if len(state_dict['state']) != len(optimizer.state):
        return False
    for param, index in pairs:
        current = optimizer.state.get(param, {})
        incoming = state_dict['state'].get(index, {})
        if current.keys() != incoming.keys():
            return False
        for key, value in current.items():
            other = incoming[key]
            if not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor):
                return False
            if (value.device != other.device or value.dtype != other.dtype
                    or value.shape != other.shape or value.stride() != other.stride()
                    or value.data_ptr() != other.data_ptr()):
                return False
    # Torch validates and restores group metadata. Retain exactly the original
    # tensor storage; native Megatron's subsequent parameter-state loader is
    # still responsible for restoring checkpoint values.
    current_state = optimizer.state
    try:
        torch.optim.Optimizer.load_state_dict(optimizer, dict(state={}, param_groups=saved))
    finally:
        optimizer.state = current_state
    return True


def install_alias_restore():
    from functools import wraps
    from transformer_engine.pytorch.optimizers import FusedAdam
    original = FusedAdam.load_state_dict
    if getattr(original, '_vine_alias_restore', False):
        return

    @wraps(original)
    def load_without_copying_same_storage(self, state_dict):
        if not restore_aliased_metadata(self, state_dict):
            return original(self, state_dict)

    load_without_copying_same_storage._vine_alias_restore = True
    FusedAdam.load_state_dict = load_without_copying_same_storage
