"""Run native critic training after collection releases the four reserved GPUs."""
import json
import os
from pathlib import Path
import subprocess
import time
import collect_supervisor as ops


def job_active(job):
    result=subprocess.run(['squeue','-h','-j',str(job),'-o','%T'],capture_output=True,text=True,check=True)
    return bool(result.stdout.strip())


def main():
    base=ops.OUT
    ops.OUT=base/'training-operations'
    ops.OUT.mkdir(parents=True,exist_ok=True)
    ops.write('supervisor.json',dict(job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        pid=os.getpid(),cgroup=Path('/proc/self/cgroup').read_text(),started=time.time()))
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
    ops.start('ray-head',ops.step(360839,'train-head',24,['bash',str(scripts/'ray_node.sh'),'head']))
    for _ in range(30):
        if ops.processes['ray-head'].poll() is not None: raise RuntimeError('Ray head exited at startup')
        result=subprocess.run(ops.step(360839,'head-check',2,['bash',str(scripts/'sif.sh'),
            'ray','status','--address=172.16.203.1:6475']),capture_output=True,text=True,timeout=120)
        (ops.OUT/'ray-head-readiness.log').write_text(result.stdout+result.stderr)
        if result.returncode==0: break
        time.sleep(5)
    else: raise RuntimeError('Ray head did not become ready')
    ops.start('ray-worker',ops.step(384912,'train-worker',24,['bash',str(scripts/'ray_node.sh'),'worker','172.16.203.1:6475']))
    for attempt in range(30):
        for name in ['ray-head','ray-worker']:
            if ops.processes[name].poll() is not None: raise RuntimeError(name+' exited at startup')
        result=subprocess.run(ops.step(360839,'ray-check',2,['bash',str(scripts/'sif.sh'),
            'ray','status','--address=172.16.203.1:6475']),capture_output=True,text=True,timeout=120)
        (ops.OUT/'ray-readiness.log').write_text(result.stdout+result.stderr)
        if result.returncode==0 and '4.0 GPU' in result.stdout: break
        time.sleep(5)
    else: raise RuntimeError('Four-GPU Ray cluster did not become ready')
    driver=ops.start('driver',ops.step(360839,'train-driver',4,['env',f'CRITIC_RUN_ROOT={base}','bash',str(scripts/'run_train.sh')]))
    while driver.poll() is None:
        if (base/'STOP').exists(): raise RuntimeError('STOP requested')
        for name in ['ray-head','ray-worker']:
            if ops.processes[name].poll() is not None: raise RuntimeError(name+' exited')
        ops.write('heartbeat.json',dict(stage='training',unix_time=time.time()))
        time.sleep(15)
    if driver.returncode: raise RuntimeError('Critic training failed')
    if not (base/'training/complete.json').exists(): raise RuntimeError('Missing training completion audit')
    ops.write('complete.json',dict(unix_time=time.time()))


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        ops.write('failed.json',dict(error=repr(exc),unix_time=time.time()));raise
    finally:
        for process in reversed(list(ops.processes.values())): ops.stop(process)
