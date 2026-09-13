"""Own existing-allocation GPU steps from a session-independent Slurm batch job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from allocation_plan import build_plan, read_allocation

P = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run-name', required=True)
parser.add_argument('--deadline', type=float)
parser.add_argument('--train-job', type=int, action='append', required=True)
parser.add_argument('--rollout-job', type=int, action='append', required=True)
parser.add_argument('--critic-job', type=int, required=True)
parser.add_argument('--resume-run', type=Path, required=True)
parser.add_argument('--stop-after-round', type=int)
args = parser.parse_args()
plan = build_plan(args.train_job, args.rollout_job, args.critic_job,
    {j:read_allocation(j) for j in [*args.train_job, *args.rollout_job]})
args.deadline = args.deadline or plan['earliest_expiry']-90*60
if not time.time() < args.deadline <= plan['earliest_expiry']-90*60:
    raise ValueError('Leave at least 90 minutes to drain and save before allocation expiry')
args.resume_run = args.resume_run.resolve()
state_dir = P/'operations'/args.run_name
state_dir.mkdir(exist_ok=False)

def write(name, obj):
    path=state_dir/name
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)

ray_labels = [f'ray-{j}' for j in [*args.train_job, *args.rollout_job]]
write('ready.json', dict(job=os.environ['SLURM_JOB_ID'], host=os.uname().nodename,
    pid=os.getpid(), cgroup=Path('/proc/self/cgroup').read_text(), started_unix=time.time(),
    plan=plan, ray_labels=ray_labels, deadline=args.deadline, resume_run=str(args.resume_run)))
# Explicit release file allows the caller to verify independence before cutover.
while not (state_dir/'GO').exists():
    time.sleep(5)

children={}
records={}
run=P/'runs'/args.run_name
python='/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python'

def start(label, command):
    log=P/'logs'/f'{args.run_name}-{label}.log'
    with log.open('x') as out:
        proc=subprocess.Popen(command, stdin=subprocess.DEVNULL,stdout=out,
            stderr=subprocess.STDOUT,start_new_session=True)
    children[label]=proc
    records[label]=dict(pid=proc.pid,command=command,log=str(log),started_unix=time.time())
    write('processes.json',records)
    return proc

def step(job, label, command, cpus=4):
    return ['srun',f'--jobid={job}','--overlap','--mem=0','--cpu-bind=none',f'--job-name={args.run_name}-{label}',
        '-N1','-n1',f'-c{cpus}',*command]

def assert_alive(labels):
    for label in labels:
        if children[label].poll() is not None:
            raise RuntimeError(f'{label} exited {children[label].returncode}; see {records[label]["log"]}')

def wait_log(label, needle, timeout=180):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        assert_alive([label])
        if needle in Path(records[label]['log']).read_text(errors='replace'):
            return
        time.sleep(5)
    raise TimeoutError(f'{label} did not become ready')

try:
    ips = {}
    for job in [*args.train_job, *args.rollout_job]:
        addresses = subprocess.check_output(step(job,'address',['hostname','-I'],cpus=1),text=True).split()
        ips[job] = next(ip for ip in addresses if ip.startswith(('172.','10.')))
    head = args.train_job[0]
    address = ips[head]+':6425'
    write('addresses.json', ips)
    environment = ['env', f'PPO_RUN_NAME={args.run_name}', f'PPO_ROLLOUT_GPUS={plan["rollout_gpus"]}',
        f'PPO_TRAIN_NODES={plan["train_nodes"]}', f'PPO_TRAIN_GPUS_PER_NODE={plan["train_gpus_per_node"]}',
        f'PPO_RESUME_RUN={args.resume_run}', f'RAY_ADDRESS={address}']
    driver_args = ['bash',str(P/'scripts/run_resume_ppo.sh'),'--offload-train',
        '--ppo-execution','overlap','--ppo-critic-equivalence-tolerance','0.01',
        '--ppo-critic-replica-host',ips[args.critic_job],
        '--ppo-critic-equivalence-contexts',str(P/'data/critic-equivalence-contexts.json'),
        '--ppo-seed-namespace','deontic-balanced-base-20260912']
    if args.stop_after_round is not None:
        driver_args += ['--ppo-stop-after-round',str(args.stop_after_round)]
    preflight = subprocess.run(step(head,'preflight',
        [*environment,*driver_args,'--ppo-preflight-only']),
        capture_output=True,text=True)
    (state_dir/'preflight.log').write_text(preflight.stdout+preflight.stderr)
    if preflight.returncode:
        raise RuntimeError('PPO resume preflight failed')
    label = f'ray-{head}'
    start(label,step(head,label,['bash',str(P/'scripts/ray_node.sh'),'train','',str(plan['train_gpus_per_node'])]))
    wait_log(label,'Ray runtime started.')
    for job in args.train_job[1:]:
        label = f'ray-{job}'
        start(label,step(job,label,['bash',str(P/'scripts/ray_node.sh'),'train_worker',address,str(plan['train_gpus_per_node'])]))
    for job in args.rollout_job:
        label = f'ray-{job}'
        start(label,step(job,label,['bash',str(P/'scripts/ray_node.sh'),'rollout',address,'2']))
    rays=list(children)
    for label in rays:
        wait_log(label,'Ray runtime started.')
    # Check Ray sees the entire resource pool before submitting placements.
    probe=step(head,'readiness',['bash',str(P/'scripts/sif.sh'),'ray','status',f'--address={address}'],cpus=2)
    deadline=time.monotonic()+180
    while True:
        assert_alive(rays)
        status=subprocess.run(probe,capture_output=True,text=True,timeout=45)
        if status.returncode==0 and f'/{plan["total_gpus"]}.0 GPU' in status.stdout:
            (state_dir/'ray-ready.txt').write_text(status.stdout)
            break
        if time.monotonic()>deadline:
            raise RuntimeError('Ray resource readiness failed: '+status.stdout+status.stderr)
        time.sleep(5)
    command=step(head,'driver',[*environment,*driver_args])
    driver=start('train',command)
    while not (run/'recipe.json').exists():
        assert_alive([*rays,'train'])
        time.sleep(5)
    start('target-audits',[python,str(P/'scripts/watch_audits.py'),str(run)])
    start('checkpoints',[python,str(P/'scripts/watch_checkpoints.py'),'--run',str(run)])
    start('evaluation',[python,str(P/'scripts/audit_evaluation.py'),'--watch-run',str(run),'--branches','1'])
    start('deadline',[python,str(P/'scripts/deadline_watch.py'),str(run),'--deadline',str(args.deadline)])
    while driver.poll() is None:
        assert_alive(rays)
        for label in ['target-audits','checkpoints','evaluation']:
            if children[label].poll() not in (None,0):
                (run/'STOP').touch()
                raise RuntimeError(f'{label} failed; requested drained stop')
        write('heartbeat.json',dict(unix_time=time.time(),driver_pid=driver.pid,
            driver_step=json.loads((run/'recipe.json').read_text())['driver_step']))
        time.sleep(15)
    if driver.returncode!=0:
        raise RuntimeError(f'Training driver exited {driver.returncode}')
    # Let the CPU audits read the final paired checkpoint after GPU work ends.
    for label in ['target-audits','checkpoints','evaluation','deadline']:
        result=children[label].wait(timeout=1800)
        if result:
            raise RuntimeError(f'{label} exited {result}')
    write('finished.json',dict(unix_time=time.time(),driver_exit=driver.returncode))
except BaseException as exc:
    write('failed.json',dict(unix_time=time.time(),error=repr(exc)))
    # On a monitor failure allow an active driver to drain before stopping workers.
    if 'train' in children and children['train'].poll() is None:
        (run/'STOP').touch()
        try:
            children['train'].wait(timeout=7200)
        except subprocess.TimeoutExpired:
            pass
    raise
finally:
    for proc in children.values():
        if proc.poll() is None:
            try:
                os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
