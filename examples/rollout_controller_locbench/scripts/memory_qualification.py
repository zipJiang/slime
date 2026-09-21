"""Isolated full-model recovery-batch replay; its optimizer updates are discarded."""
import json
from pathlib import Path
import shutil
import time


from runtime_v2 import contract, digest
from ppo_collection import audit_collection, read_rows
from ppo_protocol import lineage
from train_critic import write


def validate_source(source, resume, plan, schedule_hash):
    source = Path(source).resolve()
    if source.parent.parent != Path(resume['run']).resolve():
        raise ValueError('Qualification source must belong to the resumed run')
    start = resume['start_rollout_id']
    producer = json.loads((source.parent.parent / 'recipe.json').read_text())
    batch = json.loads((source / 'contract.json').read_text())
    if (source.name != f'train-{start:04d}'
        or producer['schedule_sha256'] != schedule_hash
        or producer['environment'] != contract()
        or batch['environment'] != contract()
        or batch['case_ids'] != plan['batches'][start]
        or batch['policy_version'] != f'actor-{start:04d}'
        or batch['value_version'] != f'critic-{start:04d}'
        or batch['actor_min_abs_advantage'] != resume['efficiency']['min_abs_advantage']
        or batch['pass_tokens'] != resume['efficiency']['pass_tokens']):
        raise ValueError('Qualification batch differs from the restored behavior or recipe')
    return source, start


def reset_peak(worker):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def memory_report(worker):
    import torch
    torch.cuda.synchronize()
    return dict(rank=worker._rank,
                total_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def configure_historical_stress(worker):
    if not worker.args.loc_memory_preflight_source:
        raise ValueError('Historical stress cannot run in production')
    worker.args.rollout_data_postprocess_path = 'memory_qualification.check_stress_likelihoods'
    # Native actors resolve this hook once during init; changing args alone
    # would leave the original production lineage checker cached here.
    worker.rollout_data_postprocess = check_stress_likelihoods


def check_stress_likelihoods(args, rollout_id, data):
    """Historical conditioning tests capacity only; no policy-equivalence claim."""
    import torch
    import torch.distributed as dist
    if not args.loc_memory_preflight_source or not args.loc_memory_stress_source:
        raise ValueError('Historical stress cannot run in production')
    current, previous = data.get('log_probs'), data.get('rollout_log_probs')
    if current is None:
        return
    differences = torch.cat([(new-old.to(new.device))[mask.to(new.device).bool()]
        for new, old, mask in zip(current, previous, data['loss_masks'], strict=True)])
    ratios = differences.exp()
    bad = torch.tensor(int(not differences.numel() or not torch.isfinite(ratios).all()),
                       device=differences.device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    write(Path(args.save).parent / 'memory-stress-audit' / f'rank-{dist.get_rank()}.json',
          dict(passed=not bool(bad.item()), scope='Historical shape stress only; updates discarded',
               source=str(args.loc_memory_stress_source), tokens=differences.numel(),
               mean_abs_difference=differences.abs().mean().item()))
    if bad.item():
        raise ValueError('Nonfinite historical stress likelihoods')


def stress_records(source, args, schedule_hash):
    source = Path(source).resolve()
    producer = json.loads((source.parent.parent/'recipe.json').read_text())
    batch = json.loads((source/'contract.json').read_text())
    if (producer['environment'] != contract() or batch['environment'] != contract()
        or producer['schedule_sha256'] != schedule_hash
        or producer['candidate_sha256'] != digest(args.loc_candidate)):
        raise ValueError('Historical stress source belongs to another experiment')
    records = [row for group in range(args.rollout_batch_size)
               for row in read_rows(source/f'group-{group:03d}.actor.jsonl.gz')]
    indices = set(sorted(range(len(records)), key=lambda i:len(records[i]['tokens']))[-4:])
    indices.update(sorted(range(len(records)), key=lambda i:records[i]['response_length'])[-4:])
    return [records[i] for i in sorted(indices)]


def qualify(args, run, resume, plan, schedule_hash, actor, critic,
            actor_batch, critic_batch, ranks):
    import ray
    source, start = validate_source(args.loc_memory_preflight_source, resume, plan, schedule_hash)
    directory = run / 'pilot-rollouts' / source.name
    shutil.copytree(source, directory, ignore=shutil.ignore_patterns(
        'exports', 'training-lineage.json', 'training-complete.json',
        'target-replay-audit.json', 'target-replay-audit.log'))
    audit_collection(directory)
    stamp = lineage(start, start, overlap=False)
    write(directory / 'training-lineage.json', stamp)
    write(run / 'status.json', dict(stage='qualifying-restored-model-memory',
                                   source=str(source), updates_will_be_discarded=True))
    records = [row for group in range(args.rollout_batch_size)
               for row in read_rows(directory / f'group-{group:03d}.actor.jsonl.gz')]
    actor_refs, selection = actor_batch(records)
    critic_refs, supervision = critic_batch(directory)
    if actor_refs is None:
        raise ValueError('Memory qualification requires actor supervision')
    reports = {}
    for role, model, refs in [('critic', critic, critic_refs), ('actor', actor, actor_refs)]:
        ray.get([h.__ray_call__.remote(reset_peak) for h in model._actor_handlers])
        began = time.monotonic()
        ray.get(model.async_train(start, refs))
        reports[role] = dict(seconds=time.monotonic()-began,
            memory=ray.get([h.__ray_call__.remote(memory_report) for h in model._actor_handlers]))
        write(run / f'memory-preflight-{role}.json', reports[role])
    audits = [json.loads((run / 'on-policy-audit' / f'round-{start:04d}-rank-{r}.json').read_text())
              for r in range(ranks)]
    if any(not row['passed'] or row['lineage'] != stamp for row in audits):
        raise ValueError('Restored-model behavior audit failed')
    if args.loc_memory_stress_source:
        historical = stress_records(args.loc_memory_stress_source, args, schedule_hash)
        refs, historical_selection = actor_batch(historical)
        if refs is None:
            raise ValueError('Historical shape stress has no supervised tokens')
        write(run/'memory-preflight-stress-source.json', dict(
            source=str(args.loc_memory_stress_source),
            source_contract_sha256=digest(args.loc_memory_stress_source/'contract.json'),
            selection=historical_selection,
            shapes=[dict(total=len(row['tokens']), response=row['response_length'],
                         supervised=sum(row['loss_mask'])) for row in historical]))
        ray.get([h.__ray_call__.remote(configure_historical_stress) for h in actor._actor_handlers])
        ray.get([h.__ray_call__.remote(reset_peak) for h in actor._actor_handlers])
        began = time.monotonic()
        ray.get(actor.async_train(start+1, refs))
        reports['historical_actor_stress'] = dict(seconds=time.monotonic()-began,
            memory=ray.get([h.__ray_call__.remote(memory_report) for h in actor._actor_handlers]))
        write(run/'memory-preflight-historical-actor.json', reports['historical_actor_stress'])
    write(run / 'memory-preflight-complete.json', dict(passed=True,
        scope='Full native restored actor/reference and critic forward/backward/optimizer on failed batch; updates discarded; no publication or checkpoint of diagnostic updates',
        source=str(source), source_contract_sha256=digest(source/'contract.json'),
        native_backend=json.loads((run/'recipe.json').read_text())['native_backend'],
        restore=json.loads((run/'initialization.json').read_text()),
        reports=reports, behavior_audits=audits, selection=selection, supervision=supervision,
        actor_updates_committed=0, critic_updates_committed=0,
        largest_sequence=max(len(row['tokens']) for row in records),
        largest_response=max(row['response_length'] for row in records)))
