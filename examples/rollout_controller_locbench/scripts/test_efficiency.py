from dataclasses import replace
from types import SimpleNamespace
import pytest
from efficiency import analyze, overlap_estimate
from ppo_batches import actor_data, actor_packets, split_mask_for_ranks
from step_controller.harness import PackedSequence
from step_controller.preparation.records import ActorSample, CriticSample, PreparedBatch


def edge(node, weight, kinds=('task',)):
    return ActorSample(node_id=node,spans=tuple(PackedSequence(tokens=(1,2,3),trainable=(False,True,True),
        logprobs={'policy':(-.1,-.2,-.3)},tag=tag) for tag in kinds),group_id=0,
        reward_config_id='r',estimator='direct_branch_td',weight=weight)


def test_cutoffs_keep_original_mass_complete_spans_and_critic():
    groups=[PreparedBatch(actor=(edge(1,.01),edge(2,-.2,('fold','task'))),
        critic=(CriticSample(node_id=0,context='private observable state',target=.5,reward_config_id='r',value_version='v'),)),
        PreparedBatch(actor=(edge(1,0),))]
    report=analyze(groups)
    full,zero,_,_,filtered=report['candidates'][:5]
    assert full['edges']==3 and full['spans']==4
    assert zero['edges']==2 and zero['actor_empty_groups']==[1]
    assert filtered['edges']==1 and filtered['spans']==2
    assert filtered['weighted_absolute_advantage_mass']==pytest.approx(.2/2/2)
    assert filtered['retained_mass_fraction']==pytest.approx(.2/.21)
    assert filtered['by_kind']['fold+task']['edges']==1
    assert report['critic_checkpoints']==1
    assert report['selection_status'].startswith('Awaiting')


def row(group=1):
    return dict(tokens=[1,2,3,4,5],response_length=4,reward=.2,loss_mask=[1,1,1,1],
        rollout_log_probs=[-.1]*4,group_index=group,
        metadata=dict(lane='actor',node_id=5,edge_tokens=4,group_edge_count=7))


def test_missing_questions_retain_original_denominator(monkeypatch):
    import ppo_batches
    def pack(args,parallel,data):
        assert args.global_batch_size==1
        assert data['rollout_mask_sums']==[28]
        return [dict(global_batch_sizes=[1])]
    monkeypatch.setattr(ppo_batches,'partition_data',pack)
    args=SimpleNamespace(global_batch_size=4)
    packets,report=actor_packets(args,{},[row()],expected_groups=range(4))
    assert packets[0]['global_batch_sizes']==[4] and args.global_batch_size==4
    assert report['empty_groups']==[0,2,3]
    packets,report=actor_packets(args,{},[],expected_groups=range(4))
    assert packets is None and report['actor_step_skipped']
    with pytest.raises(ValueError,match='unexpected'):
        actor_data([row(4)],range(4))


def test_sparse_rank_partition_preserves_every_token_coefficient_once():
    original=row();parts=split_mask_for_ranks([original],4)
    assert len(parts)==4 and original['loss_mask']==[1]*4
    data=actor_data(parts,range(4))
    assert data['rollout_mask_sums']==[28]*4
    # Arbitrary token losses show unchanged total objective, including the
    # original four-question normalizer despite three empty questions.
    token_losses=[.7,-.2,.8,1.3]
    actual=sum(sum(x*m for x,m in zip(token_losses,r['loss_mask']))/28/4 for r in parts)
    assert actual==pytest.approx(sum(token_losses)/28/4)
    assert all(sum(p['loss_mask'][i] for p in parts)==1 for i in range(4))


def test_overlap_model_exposes_only_the_slower_side():
    report=overlap_estimate(collection_seconds=30,actor_seconds=50,critic_seconds=10,publication_seconds=5)
    assert report['steady_state_seconds']==65 and report['collection_wait_seconds']==30
    assert report['learner_wait_seconds']==0
    with pytest.raises(ValueError):
        overlap_estimate(collection_seconds=float('nan'),actor_seconds=0,critic_seconds=0,publication_seconds=0)
