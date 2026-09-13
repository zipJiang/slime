"""Own fresh TRACE warmup services without touching the concurrent PPO job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import collect_supervisor as ops
from pilot_supervisor import allocated_gpus, internal_ip, job_node

operation_started = False
RETRIEVER = Path('/weka/projects/bvandur1/zjiang31/browsecomp-plus-retriever')


def interrupted(signum, frame):
    raise InterruptedError(f'Supervisor received signal {signum}')


def main():
    global operation_started
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--paired-jobs', required=True, help='Three distinct two-GPU allocations')
    p.add_argument('--single-job', type=int, required=True)
    p.add_argument('--single-gpu', type=int, default=0)
    p.add_argument('--deadline-unix', type=float, required=True)
    args = p.parse_args()
    pairs = [int(x) for x in args.paired_jobs.split(':')]
    if len(pairs) != 3 or len(set([*pairs, args.single_job])) != 4:
        raise ValueError('Expected three pairs and one separate singleton host')
    jobs = [*pairs, args.single_job]
    hosts = {str(j): job_node(j) for j in jobs}
    if len(set(hosts.values())) != 4 or any(allocated_gpus(j) < 2 for j in jobs):
        raise ValueError('Each assigned host must have two reserved GPUs')
    if args.single_gpu not in (0, 1):
        raise ValueError('Invalid singleton GPU')
    if args.deadline_unix < time.time() + 6 * 3600:
        raise ValueError('Need six hours of remaining allocation time')
    run = ops.EXPERIMENT / 'runs' / args.run_name
    storage = Path('/weka/projects/bvandur1/zjiang31/browsecomp-critic-ppo/runs') / args.run_name
    if run.exists() or run.is_symlink() or storage.exists():
        raise ValueError('Use a fresh run name')
    storage.mkdir(parents=True)
    run.symlink_to(storage, target_is_directory=True)
    ops.OUT = run
    operation_started = True
    scripts = ops.EXPERIMENT / 'scripts'
    ips = {str(j): internal_ip(j) for j in jobs}
    ops.write('supervisor.json', dict(job=os.environ['SLURM_JOB_ID'], pid=os.getpid(),
        host=os.uname().nodename, cgroup=Path('/proc/self/cgroup').read_text(),
        jobs=jobs, hosts=hosts, ips=ips, paired_jobs=pairs,
        single_gpu=args.single_gpu, deadline_unix=args.deadline_unix,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started=time.time()))
    common = ['env', f'HF_HOME={ops.CACHE}', 'HF_HUB_OFFLINE=1', 'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4']
    vllm = str(ops.EXPERIMENT.parents[2] / 'vllm/.venv/bin/vllm')
    # Pairs 0/1 are H200s. Pair 2 is the H100 host with an actual NVLink pair.
    actor_slots = [(j, gpu) for j in pairs[:2] for gpu in (0, 1)] + [(pairs[2], 0)]
    endpoints = []
    services = []
    for i, (job, gpu) in enumerate(actor_slots):
        name = f'actor-{i}'
        port = 8230 + gpu
        url = f'http://{ips[str(job)]}:{port}/v1'
        endpoints.append(url)
        ops.start(name, ops.step(job, name, 1, [*common, f'CUDA_VISIBLE_DEVICES={gpu}',
            vllm, 'serve', str(ops.BASE), '--served-model-name', 'Qwen/Qwen3.5-9B',
            '--tensor-parallel-size', '1', '--enforce-eager', '--max-model-len', '98304',
            '--enable-prefix-caching', '--mamba-cache-mode', 'all',
            '--gpu-memory-utilization', '.88', '--max-num-seqs', '16',
            '--host', '0.0.0.0', '--port', str(port)]))
        services.append((name, url + '/models'))
    judge_url = f'http://{ips[str(pairs[2])]}:8232/v1'
    ops.start('judge', ops.step(pairs[2], 'judge', 1, [*common, 'CUDA_VISIBLE_DEVICES=1',
        vllm, 'serve', str(ops.JUDGE), '--served-model-name', 'Qwen/Qwen3.5-27B',
        '--tensor-parallel-size', '1', '--enforce-eager', '--max-model-len', '8192',
        '--gpu-memory-utilization', '.88', '--max-num-seqs', '16',
        '--host', '0.0.0.0', '--port', '8232']))
    services.append(('judge', judge_url + '/models'))
    retriever_url = f'http://{ips[str(args.single_job)]}:8225'
    ops.start('retriever', ops.step(args.single_job, 'retriever', 1,
        [*common, f'CUDA_VISIBLE_DEVICES={args.single_gpu}', str(RETRIEVER / '.venv/bin/retriever'),
         'serve', '--index', str(RETRIEVER / 'indexes/qwen3-embedding-0.6b'),
         '--corpus', str(RETRIEVER / 'corpus'), '--host', '0.0.0.0', '--port', '8225',
         '--devices', 'cuda:0', '--workers', '1', '--frontend-procs', '1',
         '--max-batch-size', '512', '--max-batch-tokens', '4096', '--max-wait-ms', '10',
         '--max-length', '512', '--doc-workers', '2', '--doc-cache-gb', '8', '--doc-preload',
         '--max-inflight', '1024', '--queue-size', '4096', '--request-timeout', '30',
         '--log-level', 'warning']))
    services.append(('retriever', retriever_url + '/health'))
    for name, url in services:
        ops.ready(name, url)
    index_hashes = {f.name: hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()
                   for f in sorted((RETRIEVER / 'indexes/qwen3-embedding-0.6b').glob('*')) if f.is_file()}
    ops.write('collection/infrastructure-manifest.json', dict(
        services={name: json.loads((run / (name + '-process.json')).read_text())['command'] for name, _ in services},
        retriever_commit=subprocess.check_output(['git', '-C', str(RETRIEVER), 'rev-parse', 'HEAD'], text=True).strip(),
        retriever_client_sha256=hashlib.sha256((RETRIEVER / 'retriever/serve/client.py').read_bytes()).hexdigest(),
        index_sha256=index_hashes, jobs=jobs, actor_slots=actor_slots,
        actor_urls=endpoints, environment='trace96k', gpus=7))
    command = ['env', 'BROWSECOMP_PROFILE=trace96k', str(ops.ROOT / '.venv/bin/python'),
        str(scripts / 'trace_collect.py'), '--model', str(ops.BASE),
        '--cases', str(ops.ROOT / 'data/browsercomp-plus/cases.private.jsonl'),
        '--output', str(run / 'collection'), '--retriever-code', str(RETRIEVER),
        '--judge-url', judge_url, '--retriever-url', retriever_url, '--concurrency', '8']
    for url in endpoints:
        command.extend(['--actor-url', url])
    proc = ops.start('collect', command)
    while proc.poll() is None:
        if time.time() >= args.deadline_unix - 3600 or (run / 'STOP').exists():
            raise RuntimeError('Reached protected collection deadline or STOP request')
        for name, _ in services:
            if ops.processes[name].poll() is not None:
                raise RuntimeError(name + ' service exited')
        ops.write('heartbeat.json', dict(stage='fresh TRACE collection', unix_time=time.time()))
        time.sleep(15)
    if proc.returncode:
        raise RuntimeError('Fresh TRACE collection failed; inspect collect.log')
    for name, _ in reversed(services):
        ops.stop(ops.processes.pop(name))
    audit = ops.start('collection-audit', ['env', 'BROWSECOMP_PROFILE=trace96k',
        str(ops.ROOT / '.venv/bin/python'), str(scripts / 'trace_audit_collection.py'),
        str(run / 'collection'), '--output', str(run / 'collection-audit.json')])
    if audit.wait(timeout=1800):
        raise RuntimeError('Fresh collection readback failed')
    ops.write('collection-finished.json', dict(unix_time=time.time()))
    # The independent training supervisor consumes this boundary and owns its own services.
    ops.write('complete.json', dict(stage='collection complete; ready for critic warmup', unix_time=time.time()))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        if operation_started:
            ops.write('failed.json', dict(error=repr(exc), unix_time=time.time()))
        raise
    finally:
        for process in reversed(list(ops.processes.values())):
            ops.stop(process)
