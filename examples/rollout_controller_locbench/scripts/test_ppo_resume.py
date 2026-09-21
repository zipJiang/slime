import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import ppo_protocol as protocol
from runtime_v2 import contract,digest,MODEL


def test_resume_restores_independent_role_counts_and_rejects_untrained_cursor(tmp_path,monkeypatch):
    candidate=tmp_path/'candidate.json';candidate.write_text('{}')
    monkeypatch.setattr(protocol,'candidate',lambda _: {})
    old=tmp_path/'old';old.mkdir();(old/'checkpoints').mkdir()
    args=SimpleNamespace(loc_resume_run=old,loc_candidate=candidate,loc_allow_dp_reshard=False,
        hf_checkpoint=MODEL,ref_load=MODEL,lr=1e-6,loc_critic_lr=1e-6,global_batch_size=4,rollout_batch_size=4,
        tensor_model_parallel_size=2,pipeline_model_parallel_size=1,context_parallel_size=1,
        num_rollout=120,save_interval=6,loc_prior_strength=1.,loc_seed_namespace='seed',seq_length=98304,
        actor_num_nodes=1,actor_num_gpus_per_node=4,ckpt_format='torch_dist',use_distributed_optimizer=True,
        data_parallel_random_init=False,
        loc_pass_tokens=32768,loc_max_pass_attempts=32,loc_search_concurrency=4,
        optimizer='adam',adam_beta1=.9,adam_beta2=.95,adam_eps=1e-8,weight_decay=0,clip_grad=1,
        lr_decay_style='constant',lr_warmup_fraction=None,lr_warmup_init=0,lr_warmup_iters=0,lr_warmup_samples=0,
        eps_clip=.2,eps_clip_high=.2,eps_clip_c=None,use_kl_loss=True,kl_loss_coef=.01,
        kl_loss_type='low_var_kl',entropy_coef=0,gamma=1,lambd=1,normalize_advantages=False,
        calculate_per_token_loss=False,custom_advantage_function_path='targets.prepared_advantages',
        custom_tis_function_path='ppo_on_policy.metrics',rollout_data_postprocess_path='ppo_on_policy.check',
        use_rollout_logprobs=True,num_steps_per_rollout=1,num_critic_only_steps=0,n_samples_per_prompt=1,
        rollout_temperature=.6,rollout_top_p=.95,rollout_top_k=20,rollout_max_response_len=16384,
        sglang_context_length=98304)
    arguments={k:v for k,v in vars(args).items() if not isinstance(v,Path)}
    recipe=dict(recipe_id=protocol.RECIPE_ID,environment=contract(),candidate_sha256=digest(candidate),
        schedule_sha256='schedule',arguments=arguments,efficiency={'min_abs_advantage':.01},lineage={})
    (old/'recipe.json').write_text(json.dumps(recipe))
    evidence={}
    for role,count in [('actor',2),('critic',4)]:
        path=old/role/'iter_0000003';path.mkdir(parents=True)
        audit=path.with_name(path.name+'-readback.json')
        audit.write_text(json.dumps(dict(checkpoint=str(path),role=role,expected_optimizer_steps=count,
            optimizer_steps=[count],full_storage_read=True,finite_tensors=True)))
        evidence[str(audit.relative_to(old))]=digest(audit)
    cursor=old/'checkpoints/cursor-0003.json'
    value=dict(batches_completed=4,questions_consumed=16,schedule_sha256='schedule',prefetched_untrained_questions=0)
    cursor.write_text(json.dumps(value));evidence[str(cursor.relative_to(old))]=digest(cursor)
    record=old/'checkpoints/round-0003.json'
    boundary=dict(passed=True,iteration=3,batches_completed=4,actor_updates=2,critic_updates=4,evidence_sha256=evidence)
    record.write_text(json.dumps(boundary))
    actor,critic,_,resume=protocol.resume_roles(args,schedule_sha256='schedule')
    assert actor.start_rollout_id==critic.start_rollout_id==4 and actor.ckpt_step==critic.ckpt_step==3
    assert not actor.finetune and not critic.no_load_optim and not actor.no_load_rng
    assert resume['actor_updates']==2 and resume['critic_updates']==4
    # Recovery must not silently change the behavior/search or objective recipe.
    for name,new_value in [('loc_pass_tokens',16384),('eps_clip',.3),('kl_loss_coef',.1)]:
        previous=getattr(args,name);setattr(args,name,new_value)
        with pytest.raises(ValueError,match=name):
            protocol.resume_roles(args,schedule_sha256='schedule')
        setattr(args,name,previous)
    # A cursor containing a collected but untrained batch is invalid even if
    # its checksum is internally consistent and both model checkpoints exist.
    value['prefetched_untrained_questions']=4;cursor.write_text(json.dumps(value))
    boundary['evidence_sha256'][str(cursor.relative_to(old))]=digest(cursor);record.write_text(json.dumps(boundary))
    with pytest.raises(ValueError,match='drained'):
        protocol.resume_roles(args,schedule_sha256='schedule')
