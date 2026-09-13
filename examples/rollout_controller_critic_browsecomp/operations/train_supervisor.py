"""Run native critic training after collection releases its GPU services."""
import json
import os
from pathlib import Path
import subprocess
import time
import collect_supervisor as ops
from pilot_supervisor import allocated_gpus, internal_ip, job_node

operation_started=False


def job_active(job):
    # A completed job can age out of squeue and make -j return exit status 1.
    # Listing our active jobs distinguishes absence from a failed Slurm query.
    result=subprocess.run(['squeue','-h','-u',str(os.getuid()),'-o','%A'],
        capture_output=True,text=True,check=True)
    return str(job) in result.stdout.split()


def main():
    global operation_started
    base=ops.OUT
    training=Path(os.environ.get('CRITIC_TRAIN_OUTPUT',str(base/'training')))
    ops.OUT=Path(os.environ.get('CRITIC_TRAIN_OPERATIONS',str(base/'training-operations')))
    ops.OUT.mkdir(parents=True,exist_ok=False)
    operation_started=True
    jobs=[int(job) for job in os.environ.get('CRITIC_TRAIN_JOBS','360839:384912').split(':')]
    if len(jobs) not in (2,4) or len(set(jobs))!=len(jobs):
        raise ValueError('Expected two or four distinct two-GPU allocations')
    hosts=[job_node(job) for job in jobs]
    if len(set(hosts))!=len(jobs) or any(allocated_gpus(job)<2 for job in jobs):
        raise ValueError('Training requires two GPUs on each distinct host')
    address=internal_ip(jobs[0])+':6475'
    ops.write('supervisor.json',dict(job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        pid=os.getpid(),cgroup=Path('/proc/self/cgroup').read_text(),started=time.time(),
        gpu_jobs=jobs,training_gpus=2*len(jobs),hosts=hosts,address=address,
        training_output=str(training)))
    collection_job=json.loads((base/'supervisor.json').read_text())['job']
    until=time.monotonic()+20*3600
    while not (base/'collection-finished.json').exists():
        if (base/'STOP').exists(): raise RuntimeError('STOP requested')
        if (base/'failed.json').exists(): raise RuntimeError('Collection failed; training is not authorized by incomplete data')
        if not job_active(collection_job):
            raise RuntimeError('Collection supervisor exited without a completion marker')
        if time.monotonic()>until: raise TimeoutError('Collection deadline')
        ops.write('heartbeat.json',dict(stage='waiting for collection',unix_time=time.time()))
        time.sleep(30)
    release_deadline=time.monotonic()+10*60
    while job_active(collection_job):
        if time.monotonic()>release_deadline:
            raise RuntimeError('Collection supervisor has not released its services within ten minutes')
        time.sleep(5)
    scripts=ops.EXPERIMENT/'scripts'
    audit=ops.start('collection-audit',[str(ops.ROOT/'.venv/bin/python'),str(scripts/'audit_collection.py'),
        str(base/'collection'),'--output',str(ops.OUT/'collection-audit.json')])
    if audit.wait(timeout=1800): raise RuntimeError('Collection readback audit failed')
    ops.start('ray-head',ops.step(jobs[0],'train-head',4,['bash',str(scripts/'ray_node.sh'),'head']))
    for _ in range(30):
        if ops.processes['ray-head'].poll() is not None: raise RuntimeError('Ray head exited at startup')
        result=subprocess.run(ops.step(jobs[0],'head-check',2,['bash',str(scripts/'sif.sh'),
            'ray','status','--address='+address]),capture_output=True,text=True,timeout=120)
        (ops.OUT/'ray-head-readiness.log').write_text(result.stdout+result.stderr)
        if result.returncode==0: break
        time.sleep(5)
    else: raise RuntimeError('Ray head did not become ready')
    workers=[]
    for job in jobs[1:]:
        name=f'ray-worker-{job}'
        ops.start(name,ops.step(job,'train-worker',4,['bash',str(scripts/'ray_node.sh'),'worker',address]))
        workers.append(name)
    for attempt in range(30):
        for name in ['ray-head',*workers]:
            if ops.processes[name].poll() is not None: raise RuntimeError(name+' exited at startup')
        result=subprocess.run(ops.step(jobs[0],'ray-check',2,['bash',str(scripts/'sif.sh'),
            'ray','status','--address='+address]),capture_output=True,text=True,timeout=120)
        (ops.OUT/'ray-readiness.log').write_text(result.stdout+result.stderr)
        if result.returncode==0 and f'{2*len(jobs)}.0 GPU' in result.stdout: break
        time.sleep(5)
    else: raise RuntimeError(f'{2*len(jobs)}-GPU Ray cluster did not become ready')
    driver=ops.start('driver',ops.step(jobs[0],'train-driver',4,['env',f'CRITIC_RUN_ROOT={base}',
        f'CRITIC_TRAIN_NODES={len(jobs)}',f'CRITIC_TRAIN_OUTPUT={training}',
        f'RAY_ADDRESS={address}','bash',str(scripts/'run_train.sh')]))
    while driver.poll() is None:
        if (base/'STOP').exists(): raise RuntimeError('STOP requested')
        for name in ['ray-head',*workers]:
            if ops.processes[name].poll() is not None: raise RuntimeError(name+' exited')
        ops.write('heartbeat.json',dict(stage='training',unix_time=time.time()))
        time.sleep(15)
    if driver.returncode: raise RuntimeError('Critic training failed')
    if not (training/'complete.json').exists(): raise RuntimeError('Missing training completion audit')
    ops.write('complete.json',dict(unix_time=time.time()))


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if operation_started:
            ops.write('failed.json',dict(error=repr(exc),unix_time=time.time()))
        raise
    finally:
        for process in reversed(list(ops.processes.values())): ops.stop(process)
