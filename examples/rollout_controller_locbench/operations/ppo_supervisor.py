"""Own only LocBench PPO services; preserve paired checkpoints before lease expiry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/'scripts'))
from train_critic import write
from ppo_protocol import candidate
from collect_supervisor import step,ASSIGNMENTS,PY
from ppo_leases import refresh as refresh_leases


def reject_known_capacity(args,out,run,env):
    """Skip an impossible layout using evidence tied to this exact launch recipe."""
    proof_path=E/'operations/two-gpu-capacity.json'
    if args.train_gpus*len(args.train_jobs)!=2 or not proof_path.exists():return
    proof=json.loads(proof_path.read_text())
    for key,path in [('run_ppo_sha256',E/'scripts/run_ppo.sh'),
                     ('native_manifest_sha256',E/'operations/native-temperature-patch.json'),
                     ('candidate_sha256',args.candidate)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=proof[key]:return
    query=subprocess.run(step(args.train_jobs[0],'locbench-ppo-capacity-check',1,
        ['nvidia-smi','--query-gpu=index,memory.total','--format=csv,noheader,nounits'],gpus=max(args.assignments[args.train_jobs[0]])+1),
        env=env,capture_output=True,text=True,check=True)
    capacities={int(m[1]):int(m[2])*1024**2 for row in query.stdout.splitlines()
        if (m:=re.fullmatch(r'\s*(\d+)\s*,\s*(\d+)\s*',row))}
    if not {0,1}<=capacities.keys():raise ValueError('Missing trainer GPU capacity inventory')
    if any(proof['required_lower_bound_bytes']>capacities[d] for d in (0,1)):
        report=dict(**proof,current_capacity_bytes=capacities,models_started=0)
        write(run/'capacity-rejection.json',report)
        write(out/'failed.json',dict(time=time.time(),error='Two-GPU steady optimizer state plus retained logits exceed device capacity',evidence=str(run/'capacity-rejection.json')))
        write(out/'stopped.json',dict(time=time.time(),children={}))
        raise ValueError('Two-GPU layout is infeasible for this recipe; see capacity-rejection.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--benchmark-only',action='store_true');p.add_argument('--benchmark-source',type=Path);p.add_argument('--efficiency-plan',type=Path)
    p.add_argument('--train-gpus',type=int,choices=(2,4),default=4)
    p.add_argument('--pass-tokens',type=int,default=32768)
    p.add_argument('--train-allocator-conf',default='')
    p.add_argument('--memory-preflight-source',type=Path)
    p.add_argument('--memory-stress-source',type=Path)
    from ppo_runtime import COLLECTION_PROFILES
    p.add_argument('--collection-profile',choices=COLLECTION_PROFILES,default='original')
    p.add_argument('--critic-operations',type=Path)
    p.add_argument('--allow-profile-transition',action='store_true')
    p.add_argument('--allow-dp-reshard',action='store_true')
    p.add_argument('--profile-comparison-only',action='store_true')
    p.add_argument('--stop-after-round',type=int)
    p.add_argument('--assignments',type=Path)
    p.add_argument('--train-job',action='append',dest='train_jobs')
    p.add_argument('--replica-job',default='401540')
    args=p.parse_args()
    args.assignments=json.loads(args.assignments.read_text()) if args.assignments else ASSIGNMENTS
    args.train_jobs=args.train_jobs or ['413877']
    from ppo_layout import validate_layout,retain_launchable_rollouts
    layout=validate_layout(args.assignments,args.train_jobs,args.replica_job,args.train_gpus)
    assignments=layout['assignments'];head=args.train_jobs[0]
    critic=candidate(args.candidate)
    if critic.get('actor_identity') and args.critic_operations is None:
        raise ValueError('Imitation PPO requires its matching critic supervisor evidence')
    source=args.critic_operations or E/'operations/training-v2'
    if not (source/'stopped.json').exists() or not json.loads((source/'complete.json').read_text())['critic_ready']:
        raise ValueError('Reusable critic training must finish and release its services')
    evidence=json.loads((source/'complete.json').read_text())
    if Path(evidence['result']).resolve()!=(args.candidate.parent/'complete.json').resolve():
        raise ValueError('Critic supervisor evidence names another training run')
    out=E/'operations'/args.name;out.mkdir(exist_ok=False)
    run=E/'runs'/args.name;run.mkdir(exist_ok=False)
    children={};env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_'))}
    reject_known_capacity(args,out,run,env)
    def start(name,command):
        with (out/f'{name}.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[name]=child;write(out/f'{name}.json',dict(command=command,pid=child.pid));return child
    def interrupt(signum,frame):raise InterruptedError(signum)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    write(out/'supervisor.json',dict(job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),time=time.time()))
    try:
        leases,errors=refresh_leases(assignments,[])
        if errors:raise ValueError('PPO launch requires verified lease deadlines')
        assignments,excluded=retain_launchable_rollouts(assignments,args.train_jobs,args.replica_job,leases,10800)
        write(out/'excluded-rollout-leases.json',dict(time=time.time(),remaining_seconds=excluded))
        layout=validate_layout(assignments,args.train_jobs,args.replica_job,args.train_gpus)
        hosts={}
        for job,devices in assignments.items():
            query=subprocess.run(step(job,'locbench-ppo-host-check',1,['hostname','-I']),
                env=env,capture_output=True,text=True,check=True)
            hosts[job]=next(ip for ip in query.stdout.split() if ip.startswith('172.'))
        if len(set(hosts.values()))!=len(hosts):
            raise ValueError('Use one LocBench Ray worker per host to avoid local service collisions')
        write(out/'layout.json',dict(**layout,hosts=hosts,leases=leases))
        # Only inspect this run's selected devices, including shared allocations.
        for job,devices in assignments.items():
            query=subprocess.run(step(job,'locbench-ppo-device-check',1,
                ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                gpus=max(devices)+1),env=env,capture_output=True,text=True,check=True)
            memory={int(match[1]):int(match[2]) for row in query.stdout.splitlines()
                if (match:=re.fullmatch(r'\s*(\d+)\s*,\s*(\d+)\s*',row))}
            if not set(devices)<=memory.keys():raise ValueError(f'Missing device inventory: {job}')
            if any(memory[d]>100 for d in devices):raise ValueError(f'Assigned device still occupied: {job}: {memory}')
            write(out/f'free-{job}.json',dict(selected=devices,memory=memory))
        address=hosts[head]+':6976'
        for job in [head]+[j for j in assignments if j!=head]:
            devices=assignments[job]
            role='head' if job==head else ('train' if job in args.train_jobs else ('replica' if job==args.replica_job else 'rollout'))
            child=start('ray-'+job,step(job,'locbench-ppo-'+role,24,
                ['env','LOC_TRAIN_GPUS_PER_NODE='+str(args.train_gpus),'bash',str(E/'scripts/ppo_ray.sh'),role,address,','.join(map(str,devices))],gpus=max(devices)+1))
            if role=='head':
                deadline=time.monotonic()+300
                while 'Ray runtime started.' not in (out/('ray-'+job+'.log')).read_text(errors='replace'):
                    if child.poll() is not None:raise RuntimeError('PPO Ray head exited')
                    if time.monotonic()>deadline:raise TimeoutError('PPO Ray head startup')
                    time.sleep(5)
        deadline=time.monotonic()+600
        while True:
            if any(c.poll() is not None for c in children.values()):raise RuntimeError('PPO Ray node exited')
            check=subprocess.run(step(head,'locbench-ppo-ready-check',1,
                ['bash',str(E/'scripts/sif.sh'),'ray','status','--address='+address]),env=env,text=True,capture_output=True,timeout=90)
            (out/'readiness.log').write_text(check.stdout+check.stderr)
            if check.returncode==0 and f"/{layout['total_gpus']}.0 GPU" in check.stdout and 'locbench_ppo_replica' in check.stdout:break
            if time.monotonic()>deadline:raise TimeoutError('PPO Ray cluster startup')
            time.sleep(5)
        options=['env','RAY_ADDRESS='+address,'LOC_PPO_OUTPUT='+str(run),'LOC_CRITIC_CANDIDATE='+str(args.candidate)]
        if critic.get('actor_identity'):options+=['LOC_ACTOR_CHECKPOINT='+critic['actor_identity']['model']]
        options+=['LOC_TRAIN_NODES='+str(len(args.train_jobs)),
                  'LOC_CRITIC_REPLICA_HOST='+hosts[args.replica_job],
                  'LOC_TRAIN_GPUS_PER_NODE='+str(args.train_gpus),'LOC_ROLLOUT_GPUS='+str(layout['rollout_gpus']),
                  'LOC_PASS_TOKENS='+str(args.pass_tokens)]
        if args.train_allocator_conf:options+=['LOC_TRAIN_ALLOCATOR_CONF='+args.train_allocator_conf]
        if args.benchmark_only:options+=['LOC_PPO_BENCHMARK_ONLY=1']
        if args.benchmark_source:options+=['LOC_BENCHMARK_SOURCE='+str(args.benchmark_source)]
        if args.efficiency_plan:options+=['LOC_EFFICIENCY_PLAN='+str(args.efficiency_plan)]
        if args.resume:options+=['LOC_PPO_RESUME_RUN='+str(args.resume)]
        if args.allow_dp_reshard:options+=['LOC_ALLOW_DP_RESHARD=1']
        if args.memory_preflight_source:options+=['LOC_MEMORY_PREFLIGHT_SOURCE='+str(args.memory_preflight_source)]
        options+=['LOC_COLLECTION_PROFILE='+args.collection_profile]
        if args.allow_profile_transition:options+=['LOC_ALLOW_PROFILE_TRANSITION=1']
        if args.profile_comparison_only:options+=['LOC_PROFILE_COMPARISON_ONLY=1']
        if args.memory_stress_source:options+=['LOC_MEMORY_STRESS_SOURCE='+str(args.memory_stress_source)]
        if args.stop_after_round is not None:options+=['LOC_PPO_STOP_AFTER_ROUND='+str(args.stop_after_round)]
        driver=start('driver',step(head,'locbench-ppo-driver',8,
            options+['bash',str(E/'scripts/run_ppo.sh')],gpus=max(assignments[head])+1))
        leases=[]
        while driver.poll() is None:
            if any(c.poll() is not None for name,c in children.items() if name!='driver'):
                raise RuntimeError('A PPO Ray service exited')
            # A STOP requests a drained checkpoint in the driver; it does not
            # kill an in-flight optimizer or throw away completed trajectories.
            if (out/'STOP').exists():(run/'STOP').touch()
            leases,lease_errors=refresh_leases(assignments,leases)
            if min(v['remaining_seconds'] for v in leases)<5400:(run/'STOP').touch()
            write(out/'status.json',dict(stage='ppo',time=time.time(),leases=leases,lease_errors=lease_errors,drain_requested=(run/'STOP').exists()))
            time.sleep(30)
        if driver.returncode:raise RuntimeError('PPO driver failed; preserve and recover its latest paired boundary')
        result=run/('benchmark-complete.json' if args.benchmark_only else ('completed.json' if (run/'completed.json').exists() else 'paused.json'))
        if args.memory_preflight_source:result=run/'memory-preflight-complete.json'
        if args.profile_comparison_only:result=run/'profile-comparison-complete.json'
        write(out/'complete.json',dict(time=time.time(),result=str(result),training=json.loads(result.read_text())))
    except BaseException as exc:
        write(out/'failed.json',dict(time=time.time(),error=repr(exc)));raise
    finally:
        for child in reversed(list(children.values())):
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children.values():
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(out/'stopped.json',dict(time=time.time(),children={name:c.returncode for name,c in children.items()}))


if __name__=='__main__':main()
