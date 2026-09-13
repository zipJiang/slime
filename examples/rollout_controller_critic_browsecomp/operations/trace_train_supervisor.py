"""Train only after the full fresh TRACE collection passes readback and releases GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import collect_supervisor as ops
from pilot_supervisor import allocated_gpus, internal_ip, job_node
from train_supervisor import job_active

operation_started = False


def main():
    global operation_started
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--paired-jobs', required=True)
    p.add_argument('--single-job', type=int, required=True)
    p.add_argument('--single-gpu', type=int, default=0)
    p.add_argument('--source-candidate', type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    args = p.parse_args()
    run = ops.EXPERIMENT / 'runs' / args.run_name
    ops.OUT = run / 'training-operations'
    if not (run/'collection-finished.json').is_file() or not (run/'complete.json').is_file():
        raise ValueError('Fresh collection must finish successfully before warmup starts')
    audit = json.loads((run/'collection-audit.json').read_text())
    if not all(audit.get(k) for k in ('passed', 'full_collection', 'identities_exact')):
        raise ValueError('Full fresh collection readback failed')
    parent = json.loads((run/'supervisor.json').read_text())
    if job_active(parent['job']):
        raise RuntimeError('Collection supervisor still owns GPU services')
    pairs = [int(x) for x in args.paired_jobs.split(':')]
    if len(pairs) != 3 or len(set([*pairs, args.single_job])) != 4:
        raise ValueError('Need three GPU pairs and one separate validation host')
    if pairs != parent['paired_jobs'] or args.single_job != parent['jobs'][-1] or args.single_gpu != parent['single_gpu']:
        raise ValueError('Warmup must reuse exactly the released collection devices')
    hosts = [job_node(j) for j in [*pairs, args.single_job]]
    if len(set(hosts)) != 4 or any(allocated_gpus(j) < 2 for j in [*pairs, args.single_job]):
        raise ValueError('Warmup allocations changed')
    if args.deadline_unix < time.time()+2*3600:
        raise ValueError('Need two hours remaining for warmup and checkpoint validation')
    ops.OUT.mkdir(parents=True, exist_ok=False)
    operation_started = True
    scripts = ops.EXPERIMENT/'scripts'
    address = internal_ip(pairs[0])+':6575'
    ops.write('supervisor.json', dict(job=os.environ['SLURM_JOB_ID'], pid=os.getpid(),
        host=os.uname().nodename, cgroup=Path('/proc/self/cgroup').read_text(),
        paired_jobs=pairs, validation_job=args.single_job, validation_gpu=args.single_gpu,
        deadline_unix=args.deadline_unix, source_candidate=str(args.source_candidate), started=time.time()))
    names = []
    ops.start('ray-head', ops.step(pairs[0], 'trace-train-head', 1,
        ['bash', str(scripts/'trace_ray_node.sh'), 'head']))
    names.append('ray-head')
    until = time.monotonic()+300
    while 'Ray runtime started.' not in (ops.OUT/'ray-head.log').read_text(errors='replace'):
        if ops.processes['ray-head'].poll() is not None or time.monotonic()>until:
            raise RuntimeError('Warmup Ray head did not start')
        time.sleep(5)
    for job in pairs[1:]:
        name=f'ray-trainer-{job}'
        ops.start(name, ops.step(job, 'trace-train-worker', 1,
            ['bash', str(scripts/'trace_ray_node.sh'), 'trainer', address]))
        names.append(name)
    ops.start('ray-validator', ops.step(args.single_job, 'trace-validator', 1,
        ['bash', str(scripts/'trace_ray_node.sh'), 'validator', address, str(args.single_gpu)]))
    names.append('ray-validator')
    until=time.monotonic()+600
    while True:
        for name in names:
            if ops.processes[name].poll() is not None:
                raise RuntimeError(name+' exited at startup')
        result=subprocess.run(ops.step(pairs[0], 'trace-ray-check', 1,
            ['bash',str(scripts/'sif.sh'),'ray','status','--address='+address]),
            capture_output=True,text=True,timeout=120)
        (ops.OUT/'ray-readiness.log').write_text(result.stdout+result.stderr)
        if result.returncode == 0 and '/7.0 GPU' in result.stdout and 'browsecomp_trace_validator' in result.stdout:
            break
        if time.monotonic()>until:
            raise RuntimeError('Seven-GPU warmup cluster did not become ready')
        time.sleep(5)
    command=['env', f'CRITIC_RUN_ROOT={run}', 'CRITIC_TRAIN_NODES=3',
        f'CRITIC_TRAIN_OUTPUT={run}/training', f'RAY_ADDRESS={address}',
        f'TRACE_SOURCE_CANDIDATE={args.source_candidate.resolve()}',
        'bash', str(scripts/'run_trace_train.sh')]
    driver=ops.start('driver',ops.step(pairs[0],'trace-train-driver',1,command))
    while driver.poll() is None:
        if (run/'STOP').exists() or time.time() >= args.deadline_unix-1800:
            raise RuntimeError('Warmup reached STOP request or protected allocation deadline')
        for name in names:
            if ops.processes[name].poll() is not None:
                raise RuntimeError(name+' exited')
        ops.write('heartbeat.json',dict(stage='critic-only warmup',unix_time=time.time()))
        time.sleep(15)
    if driver.returncode or not (run/'training/complete.json').exists():
        raise RuntimeError('Warmup did not finish; inspect training/failed.json')
    ops.write('complete.json',dict(unix_time=time.time(), result=str(run/'training/complete.json'),
                                  joint_training_launched=False))


def interrupted(signum, frame):
    raise InterruptedError(f'Warmup supervisor received signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        if operation_started:
            ops.write('failed.json',dict(error=repr(exc),unix_time=time.time()))
        raise
    finally:
        for proc in reversed(list(ops.processes.values())):
            ops.stop(proc)
