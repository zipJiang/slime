"""Keep restored Adam moments on the host until the native GPU optimizer step."""
from contextlib import contextmanager


@contextmanager
def offload_actor_moments(optimizer):
    import torch
    from megatron.core.optimizer.cpu_offloading.optimizer_state_offloader import OptimizerStateOffloader

    wrappers = getattr(optimizer, 'chained_optimizers', [optimizer])
    offloaders = []
    moment_bytes = 0
    original_step = optimizer.step
    had_instance_step = 'step' in vars(optimizer)
    instance_step = vars(optimizer).get('step')
    restored = False

    def restore():
        nonlocal restored
        if not restored:
            for offloader in offloaders:
                offloader.reload()
            torch.cuda.synchronize()
            restored = True

    def step(*args, **kwargs):
        restore()
        return original_step(*args, **kwargs)

    try:
        for wrapper in wrappers:
            if getattr(wrapper, '_state_offloader', None) is not None:
                raise ValueError('Do not combine actor moment offload with native state offload')
            states = wrapper.optimizer.state
            if not states or any(not all(key in state for key in ('exp_avg', 'exp_avg_sq'))
                                 for state in states.values()):
                raise ValueError('Actor moment offload requires initialized Adam history')
            offloader = OptimizerStateOffloader(wrapper)
            offloader.mark_optimizer_states_initialized()
            moment_bytes += sum(state[key].numel()*state[key].element_size()
                                for state in states.values() for key in ('exp_avg', 'exp_avg_sq'))
            offloaders.append(offloader)
            # Keep master weights resident: native gradient preparation and
            # parameter synchronization may use them before optimizer.step().
            offloader.offload(offload_master_weights=False)
        torch.cuda.synchronize()
        for offloader in offloaders:
            offloader.release_gpu_memory()
        torch.cuda.empty_cache()
        optimizer.step = step
        yield dict(moment_bytes=moment_bytes)
    finally:
        # Restore before returning to checkpoint/export/offload lifecycle code,
        # including an early return that did not call the optimizer.
        restore()
        if had_instance_step:
            optimizer.step = instance_step
        elif 'step' in vars(optimizer):
            del optimizer.step

