"""Fail closed at a Slurm PPO migration boundary."""
import json
from pathlib import Path
import subprocess


def checkpoint_boundary(run):
    run=Path(run)
    status=json.loads((run/'paused.json').read_text())
    iterations={role:int((run/role/'latest_checkpointed_iteration.txt').read_text())
        for role in ('actor','critic')}
    if len(set(iterations.values()))!=1:
        raise ValueError('Migration source checkpoints disagree')
    iteration=iterations['actor']
    if status['completed_collection_rounds']!=iteration+1:
        raise ValueError('Migration checkpoint does not cover the paused run')
    for role in ('actor','critic'):
        count=status['completed_actor_updates'] if role=='actor' else iteration+1
        checkpoint=run/role/f'iter_{iteration:07d}'
        audit=json.loads((run/role/f'iter_{iteration:07d}-readback.json').read_text())
        if (Path(audit['checkpoint']).resolve()!=checkpoint.resolve() or audit['role']!=role
                or audit['expected_optimizer_steps']!=count or audit['optimizer_steps']!=[count]
                or not audit['full_storage_read'] or not audit['finite_tensors']):
            raise ValueError(f'Migration {role} checkpoint lacks a full matching readback')
    if not (run/'actor/rollout'/f'global_dataset_state_dict_{iteration}.pt').is_file():
        raise ValueError('Migration source lacks its question cursor')
    return dict(iteration=iteration,actor_updates=status['completed_actor_updates'],
        critic_updates=iteration+1,source=str(run.resolve()))


def migration_ready(job,run):
    try:
        output=subprocess.check_output(['sacct','-n','-X','-j',str(job),'-o','JobIDRaw,State',
            '--parsable2'],text=True,timeout=20)
    except (subprocess.SubprocessError,OSError):
        return None
    states=[line.split('|')[1].strip() for line in output.splitlines()
        if line.split('|')[0].strip()==str(job)]
    if not states or states[-1] in ('PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED'):
        return None
    if states[-1]!='COMPLETED':
        raise RuntimeError(f'Migration source supervisor {job} ended as {states[-1]}')
    return checkpoint_boundary(run)


def release_idle_allocation(job):
    try:
        steps=subprocess.check_output(['squeue','--steps','-h','-j',str(job),'-o','%i'],
            text=True,timeout=20).split()
    except (subprocess.SubprocessError,OSError):
        return False
    if any(step.rsplit('.',1)[-1] not in ('batch','extern') for step in steps):
        return False
    try:
        return subprocess.run(['scancel',str(job)],capture_output=True,timeout=20).returncode==0
    except (subprocess.SubprocessError,OSError):
        return False
