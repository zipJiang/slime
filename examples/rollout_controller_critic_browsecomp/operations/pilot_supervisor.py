"""Own a seven- or nine-GPU BrowserComp pilot across separate allocations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.request


EXPERIMENT=Path(os.environ['CRITIC_EXPERIMENT_ROOT']).resolve()
sys.path.insert(0,str(EXPERIMENT/'scripts'))
from pilot_topology import allocation_plan
CONTROLLER=EXPERIMENT.parents[2]/'rollout-controller'
CACHE=Path('/weka/projects/bvandur1/zjiang31/.cache/huggingface')
RETRIEVER=Path('/projects/bvandur1/zjiang31/browsecomp-plus-retriever')
JUDGE=CACHE/'hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654'
children={}
records={}


def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def step(job,name,cpus,command):
    return ['srun',f'--jobid={job}','--overlap','--mem=0','--cpu-bind=none',
        '-N1','-n1',f'-c{cpus}',f'--job-name=bc-pilot-{name}',*command]


def start(root,name,command):
    log=root/f'{name}.log'
    if log.exists(): raise ValueError(f'Refusing stale pilot operation log: {log}')
    with log.open('x') as stream:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
    children[name]=process
    records[name]=dict(pid=process.pid,command=command,log=str(log),
        started_unix=time.time())
    write(root,'processes.json',records)
    return process


def assert_alive(names):
    for name in names:
        if children[name].poll() is not None:
            raise RuntimeError(f'{name} exited {children[name].returncode}; inspect {records[name]["log"]}')


def job_node(job):
    result=subprocess.run(['squeue','-h','-j',str(job),'-o','%T|%N'],
        capture_output=True,text=True,check=True)
    rows=[line.strip().split('|',1) for line in result.stdout.splitlines() if line.strip()]
    if len(rows)!=1 or rows[0][0]!='RUNNING' or not rows[0][1]:
        raise ValueError(f'Pilot allocation {job} must be one running job: {rows}')
    hosts=subprocess.run(['scontrol','show','hostnames',rows[0][1]],
        capture_output=True,text=True,check=True).stdout.split()
    if len(hosts)!=1: raise ValueError(f'Pilot allocation {job} must occupy exactly one host')
    return hosts[0]


def allocated_gpus(job):
    result=subprocess.run(['scontrol','show','job','-o',str(job)],
        capture_output=True,text=True,check=True)
    match=re.search(r'\bAllocTRES=[^ ]*gres/gpu=(\d+)',result.stdout)
    if match is None: raise ValueError(f'Cannot read GPU allocation for job {job}')
    return int(match.group(1))


def internal_ip(job):
    result=subprocess.run(step(job,'address',1,['hostname','-I']),
        capture_output=True,text=True,check=True,timeout=60)
    addresses=result.stdout.split()
    private=[address for address in addresses if address.startswith(('172.','10.'))]
    if not private: raise ValueError(f'Cannot find cluster IP for allocation {job}: {addresses}')
    return private[0]


def wait_log(root,name,needle,timeout=300):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        assert_alive([name])
        if needle in (root/f'{name}.log').read_text(errors='replace'): return
        time.sleep(5)
    raise TimeoutError(f'{name} did not become ready')


def wait_http(name,url,timeout=1800,retriever=False):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        assert_alive([name])
        try:
            with urllib.request.urlopen(url,timeout=5) as response:
                value=json.load(response)
            if not retriever or (value['status']=='ok' and value['doc_workers']['ready']==2): return
        except (OSError,ValueError,KeyError): pass
        time.sleep(5)
    raise TimeoutError(f'{name} service did not become ready')


def wait_ray(root,train_job,address,names,gpus):
    probe=step(train_job,'ray-check',2,['bash',str(EXPERIMENT/'scripts/sif.sh'),
        'ray','status',f'--address={address}'])
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        assert_alive(names)
        result=subprocess.run(probe,capture_output=True,text=True,timeout=120)
        (root/'ray-readiness.log').write_text(result.stdout+result.stderr)
        if (result.returncode==0 and f'/{gpus}.0 GPU' in result.stdout
                and 'browsecomp_pilot_train' in result.stdout
                and 'browsecomp_pilot_rollout' in result.stdout
                and 'browsecomp_pilot_replica' in result.stdout): return
        time.sleep(5)
    raise RuntimeError(f'{gpus}-GPU pilot Ray resource pool did not become ready')


def stop(process):
    if process.poll() is not None: return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--train-job',type=int,action='append',required=True,
        help='one training allocation, or repeat for two two-GPU allocations')
    parser.add_argument('--train-gpus',type=int,choices=(2,4),default=4,
        help='two uses one host and DP=1; four uses DP=2 (default)')
    parser.add_argument('--inference-job',type=int,required=True,
        help='three GPUs, or two GPUs when --replica-job is supplied')
    parser.add_argument('--replica-job',type=int,
        help='optional separate allocation for the one-GPU portable critic')
    parser.add_argument('--aux-job',type=int,required=True,
        help='running single-host allocation with at least two GPUs')
    parser.add_argument('--candidate',type=Path,
        default=EXPERIMENT/'runs/base-v2/training/warmstart-candidate.json')
    parser.add_argument('--deadline-unix',type=float,required=True)
    return parser.parse_args()


def main():
    args=parse_args();run=EXPERIMENT/'runs'/args.run_name
    if run.exists() or run.is_symlink(): raise ValueError(f'Pilot run must be fresh: {run}')
    storage_root=Path(os.environ.get('PILOT_STORAGE_ROOT',
        '/weka/projects/bvandur1/zjiang31/browsecomp-critic-ppo/runs')).resolve()
    storage=storage_root/args.run_name
    storage.mkdir(parents=True,exist_ok=False);run.parent.mkdir(parents=True,exist_ok=True)
    run.symlink_to(storage,target_is_directory=True)
    free=shutil.disk_usage(storage).free;required_free=300*1024**3
    if free<required_free:
        raise RuntimeError(f'Pilot storage has {free} bytes free; {required_free} required')
    ops=run/'pilot-operations'
    ops.mkdir(parents=True)
    jobs,required=allocation_plan(args.train_job,args.inference_job,args.aux_job,args.replica_job,
        train_gpus=args.train_gpus)
    hosts={role:job_node(job) for role,job in jobs.items()}
    gpus={role:allocated_gpus(job) for role,job in jobs.items()}
    if any(gpus[role]<count for role,count in required.items()):
        raise ValueError(f'Pilot allocations lack required GPUs: allocated={gpus}, required={required}')
    if len(set(hosts.values()))!=len(jobs): raise ValueError('Pilot allocations must use distinct hosts')
    ips={role:internal_ip(job) for role,job in jobs.items()}
    if len(set(ips.values()))!=len(jobs): raise ValueError('Pilot allocation IPs are not distinct')
    if args.deadline_unix<=time.time()+6*3600:
        raise ValueError('Pilot needs at least six hours of allocation time at launch')
    retriever_tree_clean=not bool(subprocess.check_output(
        ['git','-C',str(RETRIEVER),'status','--porcelain'],text=True).strip())
    if not retriever_tree_clean: raise ValueError('Pilot retriever checkout must be clean')
    write(ops,'supervisor.json',dict(supervisor_job=os.environ['SLURM_JOB_ID'],
        pid=os.getpid(),jobs=jobs,hosts=hosts,ips=ips,gpus=gpus,required_gpus=required,
        deadline_unix=args.deadline_unix,
        candidate=str(args.candidate.resolve()),storage=str(storage),free_bytes=free,
        required_free_bytes=required_free,started_unix=time.time()))
    common=['env',f'HF_HOME={CACHE}','HF_HUB_OFFLINE=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4']
    vllm=str(EXPERIMENT.parents[2]/'vllm/.venv/bin/vllm')
    retriever_url=f'http://{ips["aux"]}:8125'
    judge_url=f'http://{ips["aux"]}:8131/v1'
    environment=['env',f'PILOT_RUN_NAME={args.run_name}',f'PILOT_CANDIDATE={args.candidate.resolve()}',
        f'PILOT_RETRIEVER_URL={retriever_url}',f'PILOT_JUDGE_URL={judge_url}',
        f'PILOT_CRITIC_REPLICA_HOST={ips.get("replica",ips["inference"])}',
        f'PILOT_TRAIN_NODES={len(args.train_job)}',f'PILOT_TRAIN_GPUS_PER_NODE={required["train"]}']
    preflight=subprocess.run([*environment,'bash',str(EXPERIMENT/'scripts/run_pilot.sh'),
        '--pilot-preflight-only'],capture_output=True,text=True)
    (ops/'preflight.log').write_text(preflight.stdout+preflight.stderr)
    if preflight.returncode: raise RuntimeError('CPU pilot preflight failed')
    start(ops,'retriever',step(args.aux_job,'retriever',20,[*common,'CUDA_VISIBLE_DEVICES=0',
        str(RETRIEVER/'.venv/bin/retriever'),'serve','--index',str(RETRIEVER/'indexes/qwen3-embedding-0.6b'),
        '--corpus',str(RETRIEVER/'corpus'),'--host','0.0.0.0','--port','8125','--devices','cuda:0',
        '--workers','1','--frontend-procs','1','--max-batch-size','512','--max-batch-tokens','4096',
        '--max-wait-ms','10','--max-length','512','--doc-workers','2','--doc-cache-gb','8',
        '--doc-preload','--max-inflight','1024','--queue-size','4096','--request-timeout','30',
        '--log-level','warning']))
    start(ops,'judge',step(args.aux_job,'judge',12,[*common,'CUDA_VISIBLE_DEVICES=1',vllm,
        'serve',str(JUDGE),'--served-model-name','Qwen/Qwen3.5-27B','--tensor-parallel-size','1',
        '--enforce-eager','--max-model-len','8192','--gpu-memory-utilization','.88',
        '--max-num-seqs','16','--host','0.0.0.0','--port','8131']))
    wait_http('retriever',retriever_url+'/health',retriever=True)
    wait_http('judge',judge_url+'/models')
    address=f'{ips["train"]}:6485'
    start(ops,'ray-train',step(jobs['train'],'ray-train',4,
        ['bash',str(EXPERIMENT/'scripts/pilot_ray_node.sh'),'train','',str(required['train'])]))
    wait_log(ops,'ray-train','Ray runtime started.')
    ray_names=['ray-train']
    for role in ['train_worker','inference','replica']:
        if role not in jobs: continue
        ray_role='rollout' if role=='inference' and 'replica' in jobs else role
        name=f'ray-{role}'
        command=['bash',str(EXPERIMENT/'scripts/pilot_ray_node.sh'),ray_role,address]
        if role=='train_worker': command.append(str(required[role]))
        start(ops,name,step(jobs[role],name,4,command))
        ray_names.append(name)
    for name in ray_names[1:]: wait_log(ops,name,'Ray runtime started.')
    wait_ray(ops,jobs['train'],address,ray_names,args.train_gpus+3)
    index=RETRIEVER/'indexes/qwen3-embedding-0.6b'
    index_hashes={}
    for path in sorted(index.glob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                index_hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    write(ops,'infrastructure-manifest.json',dict(
        schema='browsecomp-zero-warmup-pilot-infrastructure-v1',
        jobs=jobs,hosts=hosts,ips=ips,gpus=gpus,required_gpus=required,
        retriever_commit=subprocess.check_output(
            ['git','-C',str(RETRIEVER),'rev-parse','HEAD'],text=True).strip(),
        retriever_tree_clean=retriever_tree_clean,
        retriever_client_sha256=hashlib.sha256(
            (RETRIEVER/'retriever/serve/client.py').read_bytes()).hexdigest(),
        retriever_index=str(index.resolve()),retriever_index_sha256=index_hashes,
        judge_checkpoint=str(JUDGE.resolve()),
        services={name:records[name]['command'] for name in ('retriever','judge')},
        ray={name:records[name]['command'] for name in ray_names}))
    driver=start(ops,'driver',step(jobs['train'],'driver',4,[*environment,
        f'RAY_ADDRESS={address}','bash',str(EXPERIMENT/'scripts/run_pilot.sh')]))
    while driver.poll() is None:
        if time.time()>args.deadline_unix-1800:
            raise TimeoutError('Pilot reached its protected allocation deadline')
        assert_alive(['retriever','judge',*ray_names])
        write(ops,'heartbeat.json',dict(stage='pilot',unix_time=time.time()))
        time.sleep(15)
    if driver.returncode: raise RuntimeError('Pilot driver failed; inspect driver.log and run/failed.json')
    if not (run/'completed.json').is_file(): raise RuntimeError('Pilot exited without completion evidence')
    write(ops,'complete.json',dict(unix_time=time.time(),pilot=json.loads((run/'completed.json').read_text())))


if __name__=='__main__':
    ops=None
    try: main()
    except Exception as exc:
        run_name=None
        try: run_name=parse_args().run_name
        except (SystemExit,Exception): pass
        if run_name:
            root=EXPERIMENT/'runs'/run_name/'pilot-operations'
            root.mkdir(parents=True,exist_ok=True)
            write(root,'failed.json',dict(error=repr(exc),unix_time=time.time()))
        raise
    finally:
        for process in reversed(list(children.values())): stop(process)
