"""PPO roles on one four-GPU or two two-GPU training allocations."""
from datetime import datetime
import subprocess
from zoneinfo import ZoneInfo


def read_allocation(job):
    text = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True)
    fields = dict(part.split('=', 1) for part in text.split() if '=' in part)
    if fields.get('JobState') != 'RUNNING':
        raise ValueError(f'Allocation {job} is not running')
    hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', fields['NodeList']], text=True).split()
    if len(hosts) != 1:
        raise ValueError(f'Allocation {job} must occupy one host')
    tres = dict(item.split('=', 1) for item in fields['AllocTRES'].split(','))
    return dict(job=job, host=hosts[0], gpus=int(tres.get('gres/gpu', 0)),
        expires=datetime.fromisoformat(fields['EndTime']).replace(
            tzinfo=ZoneInfo('America/New_York')).timestamp())


def build_plan(train_jobs, rollout_jobs, critic_job, allocations):
    jobs = [*train_jobs, *rollout_jobs]
    if len(train_jobs) not in (1, 2) or not rollout_jobs or len(set(jobs)) != len(jobs):
        raise ValueError('Use one or two training allocations and distinct inference allocations')
    if critic_job not in rollout_jobs:
        raise ValueError('Critic inference must occupy an inference allocation')
    if len({allocations[j]['host'] for j in jobs}) != len(jobs):
        raise ValueError('PPO allocations must occupy distinct hosts')
    train_per_host = 4 // len(train_jobs)
    required = {j:train_per_host for j in train_jobs} | {j:2 for j in rollout_jobs}
    if any(allocations[j]['gpus'] < count for j, count in required.items()):
        raise ValueError('Allocation has fewer GPUs than its assigned role')
    return dict(train_jobs=train_jobs, rollout_jobs=rollout_jobs, critic_job=critic_job,
        train_nodes=len(train_jobs), train_gpus_per_node=train_per_host,
        rollout_gpus=2*len(rollout_jobs)-1, total_gpus=sum(required.values()),
        required_gpus=required, allocations=allocations,
        earliest_expiry=min(allocations[j]['expires'] for j in jobs))
