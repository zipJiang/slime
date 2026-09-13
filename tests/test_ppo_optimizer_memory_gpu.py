"""Pinned CUDA checks for restored FusedAdam moments and native memory saver."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.optimizers import FusedAdam
from torch_memory_saver import torch_memory_saver

from optimizer_memory import offload_actor_moments


@pytest.mark.parametrize('memory_saver', [False, True])
@pytest.mark.parametrize('finish', ['step', 'early_return', 'exception'])
def test_restored_adam_moments(memory_saver, finish):
    torch.manual_seed(17)
    context = torch_memory_saver.region(enable_cpu_backup=True) if memory_saver else nullcontext()
    with context:
        left = torch.nn.Parameter(torch.randn(1024, 1024, device='cuda'))
        right = torch.nn.Parameter(left.detach().clone())
        baseline = FusedAdam([left], lr=1e-6, betas=(.9, .95), weight_decay=0)
        candidate = FusedAdam([right], lr=1e-6, betas=(.9, .95), weight_decay=0)
        for _ in range(3):
            grad = torch.randn_like(left)
            left.grad, right.grad = grad, grad.clone()
            baseline.step()
            candidate.step()
        # Exercise state-dict restoration before offloading, as in the resume.
        candidate.load_state_dict(candidate.state_dict())
    wrapper = SimpleNamespace(optimizer=candidate, step=candidate.step)
    moments = {key: value.cpu().clone() for key, value in candidate.state[right].items()
               if key in ('exp_avg', 'exp_avg_sq')}
    before = torch.cuda.memory_allocated()
    error = pytest.raises(RuntimeError, match='injected') if finish == 'exception' else nullcontext()
    with error:
        with offload_actor_moments(wrapper) as report:
            assert report['moment_bytes'] == sum(x.numel()*x.element_size() for x in moments.values())
            for key in moments:
                assert candidate.state[right][key].untyped_storage().nbytes() == 0
            assert torch.cuda.memory_allocated() <= before-report['moment_bytes']
            if finish == 'exception':
                raise RuntimeError('injected')
            if finish == 'step':
                grad = torch.randn_like(left)
                left.grad, right.grad = grad, grad.clone()
                baseline.step()
                wrapper.step()
    assert wrapper.step == candidate.step
    for key in moments:
        expected = baseline.state[left][key] if finish == 'step' else moments[key]
        assert torch.equal(candidate.state[right][key].cpu(), expected.cpu())
    assert torch.equal(left, right)
    assert baseline.param_groups[0]['step'] == candidate.param_groups[0]['step']
    if memory_saver:
        torch_memory_saver.pause()
        torch_memory_saver.resume()
        for key in moments:
            expected = baseline.state[left][key] if finish == 'step' else moments[key]
            assert torch.equal(candidate.state[right][key].cpu(), expected.cpu())


def test_chained_optimizers_restore_before_native_step():
    class Chain:
        def __init__(self, optimizers):
            self.chained_optimizers = [SimpleNamespace(optimizer=o) for o in optimizers]

        def step(self):
            for wrapper in self.chained_optimizers:
                for state in wrapper.optimizer.state.values():
                    assert state['exp_avg'].untyped_storage().nbytes() > 0
                    assert state['exp_avg_sq'].untyped_storage().nbytes() > 0
                wrapper.optimizer.step()

    parameters = [torch.nn.Parameter(torch.ones(32, device='cuda')) for _ in range(2)]
    optimizers = [FusedAdam([p], lr=1e-6) for p in parameters]
    for p, optimizer in zip(parameters, optimizers):
        p.grad = torch.ones_like(p)
        optimizer.step()
    chain = Chain(optimizers)
    with offload_actor_moments(chain):
        chain.step()
    assert 'step' not in vars(chain)
    assert all(o.param_groups[0]['step'] == 2 for o in optimizers)
