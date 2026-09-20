import torch
from checkpoint_restore_memory import restore_aliased_metadata


def initialized():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    opt = torch.optim.Adam([p], lr=.1)
    p.grad = torch.tensor([.25, -.5])
    opt.step()
    return p, opt


def test_alias_restore_preserves_moments_and_restores_metadata():
    p, opt = initialized()
    old_state = opt.state
    old_moment = opt.state[p]['exp_avg']
    expected = old_moment.clone()
    saved = opt.state_dict()
    saved['param_groups'][0]['lr'] = .02
    assert restore_aliased_metadata(opt, saved)
    assert opt.state is old_state
    assert opt.state[p]['exp_avg'] is old_moment
    assert torch.equal(old_moment, expected)
    assert opt.param_groups[0]['lr'] == .02
    assert opt.param_groups[0]['params'][0] is p


def test_nonalias_values_use_normal_restore():
    p, opt = initialized()
    saved = opt.state_dict()
    saved['state'][0] = {k:v.clone() for k,v in saved['state'][0].items()}
    saved['state'][0]['exp_avg'].add_(2)
    assert not restore_aliased_metadata(opt, saved)
    opt.load_state_dict(saved)
    assert torch.equal(opt.state[p]['exp_avg'], saved['state'][0]['exp_avg'])


def test_hooked_optimizer_uses_normal_restore():
    _, opt = initialized()
    opt.register_load_state_dict_post_hook(lambda _: None)
    assert not restore_aliased_metadata(opt, opt.state_dict())
