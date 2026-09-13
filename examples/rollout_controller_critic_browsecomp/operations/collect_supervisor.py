"""Slurm-owned collection services; only these child process groups are stopped."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

EXPERIMENT=Path(os.environ['CRITIC_EXPERIMENT_ROOT'])
ROOT=EXPERIMENT.parents[2]/'rollout-controller'
OUT=EXPERIMENT/'runs'/os.environ.get('CRITIC_RUN_NAME','base-v1')
RETRIEVER=Path('/projects/bvandur1/zjiang31/browsecomp-plus-retriever')
CACHE=Path('/weka/projects/bvandur1/zjiang31/.cache/huggingface')
BASE=CACHE/'hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
JUDGE=CACHE/'hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654'
processes={}


def write(name,value):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def start(name,command):
    with (OUT/(name+'.log')).open('a') as stream:
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
    processes[name]=proc
    write(name+'-process.json',dict(pid=proc.pid,command=command,
        supervisor_job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        cgroup=Path(f'/proc/{proc.pid}/cgroup').read_text(),started=time.time()))
    return proc


def step(job,name,cpus,command):
    return ['srun',f'--jobid={job}','--overlap','--mem=0','--cpu-bind=none',
        '-N1','-n1',f'-c{cpus}',f'--job-name=bc-critic-{name}',*command]


def stop(proc):
    if proc.poll() is None:
        try: os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:
            proc.wait();return
        try: proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()


def ready(name,url):
    until=time.monotonic()+1800
    while time.monotonic()<until:
        if processes[name].poll() is not None: raise RuntimeError(name+' exited at startup')
        try:
            with urllib.request.urlopen(url,timeout=5) as r: result=json.load(r)
            if name!='retriever' or (result['status']=='ok' and result['doc_workers']['ready']==2): return
        except (OSError,ValueError,KeyError): pass
        write('heartbeat.json',dict(stage='starting '+name,unix_time=time.time()))
        time.sleep(5)
    raise TimeoutError(name+' startup timeout')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write('supervisor.json',dict(job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        pid=os.getpid(),cgroup=Path('/proc/self/cgroup').read_text(),started=time.time()))
    common=['env',f'HF_HOME={CACHE}','HF_HUB_OFFLINE=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4']
    vllm=str(EXPERIMENT.parents[2]/'vllm/.venv/bin/vllm')
    start('actor',step(384912,'actor',24,[*common,'CUDA_VISIBLE_DEVICES=0,1',vllm,'serve',str(BASE),
        '--served-model-name','Qwen/Qwen3.5-9B','--tensor-parallel-size','2','--enforce-eager',
        '--max-model-len','65536','--enable-prefix-caching','--mamba-cache-mode','all',
        '--gpu-memory-utilization','.88','--max-num-seqs','64','--host','0.0.0.0','--port','8130']))
    old=ROOT/'data/browsercomp-plus/ppo-planning-20260912/cache8'
    (old/'STOP').touch()
    until=time.monotonic()+180
    while time.monotonic()<until:
        try:
            urllib.request.urlopen('http://gh101:8125/health',timeout=3).close()
        except OSError: break
        time.sleep(5)
    else: raise RuntimeError('Previous retriever did not stop')
    # Wait for the old supervisor's srun teardown to finish before GPU profiling.
    time.sleep(10)
    start('retriever',step(360839,'retriever',20,[*common,'CUDA_VISIBLE_DEVICES=0',
        str(RETRIEVER/'.venv/bin/retriever'),'serve','--index',str(RETRIEVER/'indexes/qwen3-embedding-0.6b'),
        '--corpus',str(RETRIEVER/'corpus'),'--host','0.0.0.0','--port','8125','--devices','cuda:0',
        '--workers','1','--frontend-procs','1','--max-batch-size','512','--max-batch-tokens','4096',
        '--max-wait-ms','10','--max-length','512','--doc-workers','2','--doc-cache-gb','8',
        '--doc-preload','--max-inflight','1024','--queue-size','4096','--request-timeout','30',
        '--log-level','warning']))
    start('judge',step(360839,'judge',12,[*common,'CUDA_VISIBLE_DEVICES=1',vllm,'serve',str(JUDGE),
        '--served-model-name','Qwen/Qwen3.5-27B','--tensor-parallel-size','1','--enforce-eager',
        '--max-model-len','8192','--gpu-memory-utilization','.88','--max-num-seqs','16',
        '--host','0.0.0.0','--port','8131']))
    for name,url in [('actor','http://gh129:8130/v1/models'),('retriever','http://gh101:8125/health'),
                     ('judge','http://gh101:8131/v1/models')]: ready(name,url)
    hashes={}
    for path in sorted((RETRIEVER/'indexes/qwen3-embedding-0.6b').glob('*')):
        if path.is_file():
            with path.open('rb') as stream: hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    write('collection/infrastructure-manifest.json',dict(
        retriever_commit=subprocess.check_output(['git','-C',str(RETRIEVER),'rev-parse','HEAD'],text=True).strip(),
        retriever_client_sha256=hashlib.sha256((RETRIEVER/'retriever/serve/client.py').read_bytes()).hexdigest(),
        index_sha256=hashes,services={name:json.loads((OUT/(name+'-process.json')).read_text())['command']
            for name in ['actor','judge','retriever']}))
    command=[str(ROOT/'.venv/bin/python'),str(EXPERIMENT/'scripts/collect.py'),
        '--model',str(BASE),'--cases',str(ROOT/'data/browsercomp-plus/cases.private.jsonl'),
        '--output',str(OUT/'collection'),'--retriever-code',str(RETRIEVER),
        '--actor-url','http://gh129:8130/v1','--judge-url','http://gh101:8131/v1',
        '--retriever-url','http://gh101:8125','--concurrency','24']
    # Pilot and full collection share immutable protocol and deterministic IDs.
    stages=[('collect',[])] if os.environ.get('CRITIC_SKIP_PILOT')=='1' else [('pilot',['--pilot']),('collect',[])]
    for stage,extra in stages:
        proc=start(stage,command+extra)
        while proc.poll() is None:
            if (OUT/'STOP').exists(): raise RuntimeError('STOP requested')
            for name in ['actor','retriever','judge']:
                if processes[name].poll() is not None: raise RuntimeError(name+' exited')
            write('heartbeat.json',dict(stage=stage,unix_time=time.time()))
            time.sleep(15)
        if proc.returncode: raise RuntimeError(stage+' failed; inspect '+stage+'.log')
    write('collection-finished.json',dict(unix_time=time.time()))


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        write('failed.json',dict(error=repr(exc),unix_time=time.time()));raise
    finally:
        for proc in reversed(list(processes.values())): stop(proc)
