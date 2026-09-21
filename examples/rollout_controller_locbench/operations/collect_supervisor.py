"""Durable ownership of LocBench warmup servers and collector; retains reservations."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from slurm_retry import wrap

E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/'scripts'))
from collect_warmup import write
from runtime import MODEL

PY='/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python'
ASSIGNMENTS={'413877':[0,1,2,3],'384963':[0,1],'384964':[0,1],'360840':[0,1],
    '409247':[0,1],'412882':[0,1],'401540':[1]}


def step(job,name,cpus,command,gpus=0):
    return wrap(['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(cpus),'--cpu-bind=none','--mem=0',
        '--gres='+('gpu:'+str(gpus) if gpus else 'none'),'--job-name='+name,
        '--chdir='+str(E),*command])


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--pilot',type=int,default=0)
    args=p.parse_args();out=E/'operations'/args.name;out.mkdir(exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_'))}
    children={}
    def start(name,cmd):
        with (out/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[name]=proc;write(out/f'{name}.json',dict(pid=proc.pid,command=cmd));return proc
    def interrupted(signum,frame):raise InterruptedError(signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    write(out/'supervisor.json',dict(job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),assignments=ASSIGNMENTS,time=time.time()))
    try:
        for job,devices in ASSIGNMENTS.items():
            start('servers-'+job,step(job,'locbench-mc-servers',max(8,4*len(devices)),
                [PY,str(E/'operations/serve.py'),'--devices',','.join(map(str,devices)),
                    '--output',str(out/('pool-'+job))],gpus=4 if job=='413877' else 2))
        deadline=time.monotonic()+1500
        while not all((out/('pool-'+j)/'ready.json').exists() for j in ASSIGNMENTS):
            if any(c.poll() is not None for c in children.values()):raise RuntimeError('A server step exited')
            if time.monotonic()>deadline:raise TimeoutError('Pool startup')
            write(out/'status.json',dict(stage='starting-servers',time=time.time()));time.sleep(10)
        urls=[url for job in ASSIGNMENTS for url in json.loads((out/('pool-'+job)/'ready.json').read_text())['urls']]
        write(out/'servers.json',dict(model=MODEL,urls=urls,assignments=ASSIGNMENTS))
        command=[PY,'-u',str(E/'scripts/collect_warmup.py'),'--servers',str(out/'servers.json'),
            '--output',str(E/'runs/base-critic-v1/collection'),'--pilot',str(args.pilot)]
        collector=start('collector',step('413877','locbench-critic-collection',8,command))
        while collector.poll() is None:
            if any(c.poll() is not None for name,c in children.items() if name!='collector'):
                raise RuntimeError('Server step exited during collection')
            if (out/'STOP').exists():raise InterruptedError('STOP requested')
            write(out/'status.json',dict(stage='collecting',time=time.time()));time.sleep(15)
        if collector.returncode:raise RuntimeError(f'Collector exited {collector.returncode}')
        if args.pilot:
            check=subprocess.run(step('413877','locbench-mc-audit',4,[PY,str(E/'scripts/audit_warmup.py'),
                '--collection',str(E/'runs/base-critic-v1/collection'),'--expected',str(args.pilot)]),env=env)
            if check.returncode:raise RuntimeError('Pilot native readback failed')
            full_command=command[:-2]
            collector=start('full-collector',step('413877','locbench-critic-collection',8,full_command))
            while collector.poll() is None:
                if any(c.poll() is not None for name,c in children.items() if name.startswith('servers-')):
                    raise RuntimeError('Server step exited during collection')
                if (out/'STOP').exists():raise InterruptedError('STOP requested')
                write(out/'status.json',dict(stage='full-collection',time=time.time()));time.sleep(15)
            if collector.returncode:raise RuntimeError('Full collection failed')
        check=subprocess.run(step('413877','locbench-mc-audit',4,[PY,str(E/'scripts/audit_warmup.py'),
            '--collection',str(E/'runs/base-critic-v1/collection'),'--expected','320','--complete']),env=env)
        if check.returncode:raise RuntimeError('Full collection native readback failed')
        write(out/'complete.json',dict(time=time.time(),pilot=args.pilot))
    except BaseException as exc:
        write(out/'failed.json',dict(error=repr(exc),time=time.time()));raise
    finally:
        for job in ASSIGNMENTS:
            pool=out/('pool-'+job)
            if pool.exists():(pool/'STOP').touch()
        # Each server wrapper owns and drains only its own children.
        for name,c in reversed(list(children.items())):
            if 'collector' in name and c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
            try:c.wait(timeout=55)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid,signal.SIGTERM)
                try:c.wait(timeout=35)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
        write(out/'stopped.json',dict(time=time.time(),children={n:c.returncode for n,c in children.items()}))


if __name__=='__main__':main()
