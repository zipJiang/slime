"""Describe the same nine pilot GPUs on three to five separate allocations."""


def required_gpus(roles):
    roles=set(roles)
    if not {'train','inference','aux'} <= roles or roles-{'train','train_worker','inference','replica','aux'}:
        raise ValueError('Unknown or incomplete pilot allocation roles')
    required=dict(train=2 if 'train_worker' in roles else 4,
                  inference=2 if 'replica' in roles else 3,aux=2)
    if 'train_worker' in roles: required['train_worker']=2
    if 'replica' in roles: required['replica']=1
    return required


def allocation_plan(train_jobs,inference_job,aux_job,replica_job=None):
    if len(train_jobs) not in (1,2):
        raise ValueError('Use one four-GPU or two two-GPU training allocations')
    jobs=dict(train=train_jobs[0],inference=inference_job,aux=aux_job)
    if len(train_jobs)==2: jobs['train_worker']=train_jobs[1]
    if replica_job is not None: jobs['replica']=replica_job
    if len(set(jobs.values()))!=len(jobs):
        raise ValueError('Pilot allocation roles must use distinct jobs')
    return jobs,required_gpus(jobs)
