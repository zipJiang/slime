"""LocBench question schedule, frozen behavior lineage and reusable-critic gate."""
import copy
import hashlib
import json
from pathlib import Path
from runtime_active import EXPERIMENT, MODEL, contract, digest

from ppo_runtime import transition,validate_environment

RECIPE_ID='locbench-base-pretrained-critic-ppo-v1'


def schedule(split, *, updates=120, batch_size=4, warmup_ids=()):
    train=list(split['train']);held=set(split['development'])|set(split['test'])
    if (type(updates) is not int or updates<2 or type(batch_size) is not int or not 1<=batch_size<=len(train)
        or len(set(train))!=len(train) or set(train)&held or not set(warmup_ids)<=set(train)):
        raise ValueError('Invalid training schedule or overlapping split')
    questions=[];epoch=0
    while len(questions)<updates*batch_size:
        tail=set(questions[-(len(questions)%batch_size):]) if len(questions)%batch_size else set()
        order=sorted(train,key=lambda q:(q in tail,q in set(warmup_ids) if epoch==0 else False,
            hashlib.sha256(f'{RECIPE_ID}/{epoch}/{q}'.encode()).hexdigest()))
        questions.extend(order);epoch+=1
    questions=questions[:updates*batch_size]
    batches=[questions[i:i+batch_size] for i in range(0,len(questions),batch_size)]
    if any(len(set(batch))!=batch_size for batch in batches):raise ValueError('Duplicate question within batch')
    return dict(schema=RECIPE_ID,updates=updates,batch_size=batch_size,questions=questions,batches=batches,
        training_pool=train,full_dataset_questions=len(train),
        split_sha256=hashlib.sha256(json.dumps(split,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def lineage(collection_round,behavior_round,*,overlap):
    lag=collection_round-behavior_round
    if collection_round<0 or behavior_round<0 or not 0<=lag<=(1 if overlap else 0):
        raise ValueError('Behavior snapshot exceeds the permitted one-batch lag')
    return dict(collection_round=collection_round,behavior_round=behavior_round,learner_round=collection_round,
        actor_lag=lag,critic_lag=lag,denominator='stored_behavior_logprobs',execution='overlap' if overlap else 'sync')


def validate_lineage(value,round_id):
    if value!=lineage(round_id,value['behavior_round'],overlap=value['execution']=='overlap'):
        raise ValueError('Frozen batch lineage changed')
    return value['actor_lag']


def boundary(round_id,*,save_interval,last_round,stop=False):
    if save_interval<1:raise ValueError('Positive save interval required')
    return stop or round_id in (0,1,last_round) or (round_id+1)%save_interval==0


def prefetch(round_id,*,last_round,save_now,stop=False):
    return round_id>=2 and round_id<last_round and not save_now and not stop


def candidate(path):
    path=Path(path).resolve();value=json.loads(path.read_text())
    if not value.get('ready') or value['selected_update']<=0 or value['native_iteration']!=value['selected_update']-1:
        raise ValueError('No improving reusable critic selected')
    expected=contract()
    if value.get('actor_identity') is not None:
        from imitation_lineage import require_model
        from synthetic_runtime import environment
        identity=require_model(value['actor_identity']['training'])
        if identity!=value['actor_identity'] or value.get('model_initialization')!=identity['model']:
            raise ValueError('Critic was not initialized from the verified imitation actor')
        expected=environment()
    if value['environment']!=expected or value['context_source_sha256']!=digest(EXPERIMENT/'scripts/runtime_v2.py'):
        raise ValueError('PPO environment/context differs from critic pretraining')
    best=value['selected'];initial=value['initial']
    if not best['selection_mse']<min(initial['selection_mse'],best['selection_baseline_mse']):
        raise ValueError('Selected critic does not beat both baselines')
    audit=path.parent/'reload-audit.json'
    if digest(audit)!=value['validation_audit_sha256'] or not json.loads(audit.read_text())['passed']:
        raise ValueError('Critic model-only reload audit is missing or changed')
    native=Path(value['native_checkpoint'])/f"iter_{value['native_iteration']:07d}"
    evidence=native.with_name(native.name+'-readback.json')
    report=json.loads(evidence.read_text())
    if (Path(report['checkpoint']).resolve()!=native.resolve() or report['role']!='critic'
        or report['expected_optimizer_steps']!=value['selected_update']
        or report['optimizer_steps']!=[value['selected_update']]
        or not report['full_storage_read'] or not report['finite_tensors']):
        raise ValueError('Native critic checkpoint lacks matching optimizer/tensor evidence')
    if not native.is_dir() or not Path(value['inference']).is_dir():raise ValueError('Selected critic storage is missing')
    return value


def initial_roles(args,candidate_path):
    value=candidate(candidate_path)
    from ppo_runtime import environment
    model=value['actor_identity']['model'] if value.get('actor_identity') else MODEL
    if any(Path(getattr(args,key)).resolve()!=Path(model).resolve() for key in ('hf_checkpoint','load','ref_load')):
        raise ValueError('Actor and reference must match the critic collection actor')
    profile=getattr(args,'loc_collection_profile','original')
    profile_transition=transition(value['environment'],profile,
        allow=getattr(args,'loc_allow_profile_transition',False))
    if args.num_critic_only_steps!=0 or args.start_rollout_id not in (None,0) or args.ckpt_step is not None:
        raise ValueError('Fresh PPO starts at zero with no additional critic warmup')
    actor,critic=copy.deepcopy(args),copy.deepcopy(args)
    for role in (actor,critic):
        role.finetune=True;role.no_load_optim=True;role.no_load_rng=True;role.start_rollout_id=0
    critic.load=value['native_checkpoint'];critic.ckpt_step=value['native_iteration']
    return actor,critic,dict(actor_base=model,actor_identity=value.get('actor_identity'),critic_candidate=str(Path(candidate_path).resolve()),
        candidate_sha256=digest(candidate_path),critic_native_checkpoint=critic.load,critic_native_iteration=critic.ckpt_step,
        profile_transition=profile_transition,
        initialization='Matched actor/reference; model-only selected critic; fresh optimizers and question cursor')


def resume_roles(args,*,schedule_sha256):
    root=Path(args.loc_resume_run).resolve()
    recipe=json.loads((root/'recipe.json').read_text())
    value=candidate(args.loc_candidate)
    validate_environment(recipe)
    profile_transition=transition(recipe['environment'],getattr(args,'loc_collection_profile','original'),
        allow=getattr(args,'loc_allow_profile_transition',False))
    if (recipe['recipe_id']!=RECIPE_ID
        or recipe['candidate_sha256']!=digest(args.loc_candidate) or recipe['schedule_sha256']!=schedule_sha256):
        raise ValueError('Resume changes the model, critic, environment or question schedule')
    previous=recipe['arguments']
    fixed=('hf_checkpoint','ref_load','lr','loc_critic_lr','global_batch_size','rollout_batch_size',
           'tensor_model_parallel_size','pipeline_model_parallel_size','context_parallel_size',
           'num_rollout','save_interval','loc_prior_strength','loc_seed_namespace','seq_length',
           'loc_pass_tokens','loc_max_pass_attempts','loc_search_concurrency',
           'optimizer','adam_beta1','adam_beta2','adam_eps','weight_decay','clip_grad','lr_decay_style',
           'lr_warmup_fraction','lr_warmup_init','lr_warmup_iters','lr_warmup_samples',
           'eps_clip','eps_clip_high','eps_clip_c','use_kl_loss','kl_loss_coef','kl_loss_type','entropy_coef',
           'gamma','lambd','normalize_advantages','calculate_per_token_loss',
           'custom_advantage_function_path','custom_tis_function_path','rollout_data_postprocess_path',
           'use_rollout_logprobs','num_steps_per_rollout','num_critic_only_steps','n_samples_per_prompt',
           'rollout_temperature','rollout_top_p','rollout_top_k','rollout_max_response_len',
           'sglang_context_length')
    for key in fixed:
        if previous.get(key)!=getattr(args,key):raise ValueError('Resume changes '+key)
    shape=('actor_num_nodes','actor_num_gpus_per_node')
    changed=any(previous.get(k)!=getattr(args,k) for k in shape)
    if changed and not args.loc_allow_dp_reshard:raise ValueError('Trainer topology change requires explicit DP reshard')
    if any(getattr(args,k)!=v or previous.get(k)!=v for k,v in
           dict(tensor_model_parallel_size=2,pipeline_model_parallel_size=1,context_parallel_size=1,
                ckpt_format='torch_dist',use_distributed_optimizer=True,data_parallel_random_init=False).items()):
        raise ValueError('Native resume requires unchanged model sharding and distributed optimizer')
    selected=None
    for path in sorted((root/'checkpoints').glob('round-*.json'),reverse=True):
        row=json.loads(path.read_text());iteration=row['iteration']
        if not row.get('passed') or row['batches_completed']!=iteration+1:continue
        required={f'{role}/iter_{iteration:07d}-readback.json' for role in ('actor','critic')}
        required.add(f'checkpoints/cursor-{iteration:04d}.json')
        if set(row['evidence_sha256'])!=required:raise ValueError('Incomplete paired checkpoint evidence')
        for name,sha in row['evidence_sha256'].items():
            evidence=(root/name).resolve()
            if not evidence.is_relative_to(root) or digest(evidence)!=sha:raise ValueError('Changed checkpoint evidence')
        if not all((root/role/f'iter_{iteration:07d}').is_dir() for role in ('actor','critic')):continue
        cursor=json.loads((root/f'checkpoints/cursor-{iteration:04d}.json').read_text())
        if (cursor['batches_completed']!=iteration+1 or cursor['questions_consumed']!=(iteration+1)*args.rollout_batch_size
            or cursor['schedule_sha256']!=schedule_sha256 or cursor['prefetched_untrained_questions']!=0):
            raise ValueError('Saved question cursor is not a drained training boundary')
        for role in ('actor','critic'):
            audit=json.loads((root/role/f'iter_{iteration:07d}-readback.json').read_text())
            if (audit['role']!=role or audit['expected_optimizer_steps']!=row[f'{role}_updates']
                or audit['optimizer_steps']!=[row[f'{role}_updates']] or not audit['full_storage_read']
                or not audit['finite_tensors'] or Path(audit['checkpoint']).resolve()!=(root/role/f'iter_{iteration:07d}').resolve()):
                raise ValueError('Native role checkpoint/counter mismatch')
            if changed and audit['common_state'].get('optimizer',{}).get('param_state_sharding_type')!='dp_reshardable':
                raise ValueError('Checkpoint optimizer cannot be resharded')
        selected=row;break
    if selected is None:raise ValueError('No complete paired checkpoint to resume')
    start=selected['batches_completed'];actor,critic=copy.deepcopy(args),copy.deepcopy(args)
    for role,target in [('actor',actor),('critic',critic)]:
        target.load=str(root/role);target.ckpt_step=start-1;target.start_rollout_id=start
        target.finetune=False;target.no_load_optim=False;target.no_load_rng=False
    resume=dict(run=str(root),start_rollout_id=start,iteration=start-1,
        actor_updates=selected['actor_updates'],critic_updates=selected['critic_updates'],
        efficiency=recipe['efficiency'],topology_changed=changed,profile_transition=profile_transition)
    return actor,critic,recipe['lineage'],resume
