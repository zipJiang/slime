import pytest

from critic_selection import learned_critic_rows


def row(group,terminal,target=0.,mean=0.):
    return dict(group_index=group,target=target,metadata=dict(lane='critic',
        terminal_boundary=terminal,diagnostics=dict(mean_return=mean)))


def test_terminal_boundaries_are_excluded_from_learned_values():
    selected,report=learned_critic_rows([row(0,False,.5,.5),row(0,True),row(1,False,1.,1.)])
    assert len(selected)==2
    assert report['learned']==2 and report['fixed_terminal']==1


def test_nonzero_terminal_future_return_is_rejected():
    with pytest.raises(ValueError,match='zero future'):
        learned_critic_rows([row(0,False,.5,.5),row(0,True,1.,1.)])


def test_each_question_requires_a_nonterminal_target():
    with pytest.raises(ValueError,match='Every question'):
        learned_critic_rows([row(0,False,.5,.5),row(1,True)])


def test_boundary_flag_is_required():
    bad=row(0,False);del bad['metadata']['terminal_boundary']
    with pytest.raises(ValueError,match='boundary'):
        learned_critic_rows([bad])
