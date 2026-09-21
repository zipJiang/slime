"""Validate dedicated LocBench PPO GPU ownership before starting services."""
def validate_layout(assignments,train_jobs,replica_job,train_gpus):
    if not isinstance(assignments,dict) or not assignments:
        raise ValueError('A nonempty job-to-device assignment is required')
    for job,devices in assignments.items():
        if not isinstance(job,str) or not job.isdigit() or not isinstance(devices,list) or not devices:
            raise ValueError('Each numeric job must own a nonempty device list')
        if any(type(d) is not int or d<0 for d in devices) or len(set(devices))!=len(devices):
            raise ValueError('Device indices must be unique nonnegative integers')
    if not train_jobs or len(set(train_jobs))!=len(train_jobs) or any(j not in assignments for j in train_jobs):
        raise ValueError('Training jobs must be distinct assigned jobs')
    if replica_job not in assignments or replica_job in train_jobs or len(assignments[replica_job])!=1:
        raise ValueError('The critic replica requires one separately assigned GPU')
    if train_gpus not in (2,4) or len(train_jobs)*train_gpus not in (2,4):
        raise ValueError('Native PPO supports two or four training GPUs in local TP2 pairs')
    if any(len(assignments[j])<train_gpus for j in train_jobs):
        raise ValueError('Training host lacks its requested GPU count')
    total=sum(map(len,assignments.values()))
    rollout=total-len(train_jobs)*train_gpus-1
    if rollout<2:raise ValueError('PPO requires at least two rollout GPUs')
    return dict(assignments=assignments,train_jobs=train_jobs,replica_job=replica_job,
                train_gpus_per_node=train_gpus,total_gpus=total,rollout_gpus=rollout)


def retain_launchable_rollouts(assignments,train_jobs,replica_job,leases,minimum_seconds):
    """Expire optional rollout hosts without changing trainer or replica ownership."""
    remaining={row['job']:row['remaining_seconds'] for row in leases}
    if set(remaining)!=set(assignments):
        raise ValueError('Lease inventory differs from GPU assignments')
    required=set(train_jobs)|{replica_job}
    if any(remaining[j]<minimum_seconds for j in required):
        raise ValueError('Training and critic replica leases are too short for PPO launch')
    kept={j:ds for j,ds in assignments.items() if j in required or remaining[j]>=minimum_seconds}
    excluded={j:remaining[j] for j in assignments if j not in kept}
    return kept,excluded
