"""Move only an explicitly named experiment pool at a verified checkpoint.

The config names owned Slurm *steps*, never allocations. The original PPO pool
is outside this list. This helper launches the synchronous benchmark arm; later
promotion requires benchmark_compare.py and a separate explicit launch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def live(step):
    result = subprocess.run(['squeue', '--steps='+step, '-h', '-o', '%i'],
                            capture_output=True, text=True, check=True)
    return step in result.stdout.split()


def srun(job, *command, cpus=32):
    return ['srun', f'--jobid={job}', '--overlap', '--cpu-bind=none', '-N1', '-n1', f'-c{cpus}', *command]


def launch(command, log):
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return dict(pid=process.pid, command=command, log=str(log))


def main(config_path, journal):
    config = read(config_path)
    source, scripts = Path(config['source_run']), Path(__file__).resolve().parent
    recipe = read(source/'recipe.json')
    steps = [config['source_driver'], *[node['ray_step'] for node in config['nodes']]]
    if recipe['driver_step'] != config['source_driver']:
        raise ValueError('Source driver ownership mismatch')
    if any(len(step.split('.')) != 2 or not all(p.isdigit() for p in step.split('.'))
           or step.endswith('.0') for step in steps):
        raise ValueError('Only explicit non-interactive experiment steps may be stopped')
    if len(set(steps)) != len(steps):
        raise ValueError('Duplicate owned step')
    for node in config['nodes']:
        if not node['ray_step'].startswith(str(node['job'])+'.'):
            raise ValueError('Ray step belongs to another allocation')
    iteration = config['checkpoint_iteration']
    state = dict(stage='waiting_for_verified_checkpoint', config=config,
        started_at=datetime.now(timezone.utc).isoformat(), actions=[])
    write(journal, state)
    while True:
        if not live(config['source_driver']):
            raise ValueError('Source driver ended before the planned transition')
        status = read(source/'status.json')
        if status['completed_collection_rounds'] > iteration+1:
            raise ValueError('Source has advanced beyond the planned boundary')
        ready = status['completed_collection_rounds'] == iteration+1
        for role in ('actor','critic'):
            audit_path = source/role/f'iter_{iteration:07d}-readback.json'
            tracker = source/role/'latest_checkpointed_iteration.txt'
            if not audit_path.exists() or not tracker.exists():
                ready = False
                continue
            try:
                audit = read(audit_path)
            except json.JSONDecodeError:
                ready = False
                continue  # The readback writer has not closed the JSON yet.
            expected = max(0, iteration+1-recipe['arguments']['num_critic_only_steps']) if role == 'actor' else iteration+1
            if not (int(tracker.read_text()) == iteration and audit['role'] == role and
                    audit['full_storage_read'] and audit['finite_tensors'] and
                    audit['optimizer_steps'] == [expected] and
                    Path(audit['checkpoint']).resolve() == (source/role/f'iter_{iteration:07d}').resolve()):
                raise ValueError(f'Invalid {role} resume audit')
        cursor = source/'actor/rollout'/f'global_dataset_state_dict_{iteration}.pt'
        ready = ready and cursor.exists()
        if ready:
            break
        time.sleep(20)
    # Store the exact recovery evidence before terminating any process.
    state.update(stage='stopping_owned_steps', source_status=status,
        cursor_sha256=hashlib.sha256(cursor.read_bytes()).hexdigest())
    write(journal, state)
    write(source/'paused-for-benchmark.json', dict(journal=str(journal), checkpoint=iteration,
          source_status=status, reason='Paired throughput benchmark; resume state verified before transition'))
    for step in steps:
        if live(step):
            subprocess.run(['scancel', step], check=True)
            state['actions'].append(dict(action='stop_experiment_step', step=step,
                                        time=datetime.now(timezone.utc).isoformat()))
            write(journal, state)
    deadline = time.monotonic()+180
    while any(live(step) for step in steps):
        if time.monotonic() > deadline:
            raise TimeoutError('Owned Slurm steps did not terminate')
        time.sleep(5)
    # Verify the pool is clear before the new Ray placement reserves its GPUs.
    for node in config['nodes']:
        result = subprocess.run(srun(node['job'], 'nvidia-smi',
            '--query-compute-apps=pid', '--format=csv,noheader', cpus=1),
            capture_output=True, text=True, check=True)
        if any(line.strip().isdigit() for line in result.stdout.splitlines()):
            raise RuntimeError(f"GPU processes remain on {node['host']}; refusing to compete")
    logs = journal.parent/'logs'
    logs.mkdir(exist_ok=True)
    processes = []
    for node in config['nodes']:
        command = srun(node['job'], 'bash', str(scripts/'ray_node.sh'), node['role'])
        if node['role'] != 'train':
            command += [config['ray_address'], str(node['gpus'])]
        processes.append(dict(host=node['host'], **launch(command, logs/f"ray-{node['host']}.log")))
        if node['role'] == 'train':
            # Workers retry registration, but the head must have time to bind.
            time.sleep(10)
    state.update(stage='launching_sync_benchmark', ray_processes=processes)
    write(journal, state)
    time.sleep(10)
    train_node = next(node for node in config['nodes'] if node['role'] == 'train')
    command = srun(train_node['job'], 'env', f"PPO_RUN_NAME={config['run_name']}",
        f'PPO_RESUME_RUN={source}', f"RAY_ADDRESS={config['ray_address']}",
        'PPO_ROLLOUT_GPUS=6', 'bash', str(scripts/'run_resume_ppo.sh'),
        '--ppo-execution', 'sync', '--ppo-benchmark', '--ppo-stop-after-round',
        str(iteration+config['measured_updates']), '--ppo-seed-namespace', config['seed_namespace'])
    state.update(stage='sync_benchmark_launched', driver=launch(command, logs/'sync-driver.log'))
    write(journal, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.config.resolve(), args.journal.resolve())
    except Exception as exc:
        error = args.journal.with_name(args.journal.stem+'-failed.json')
        write(error, dict(error=repr(exc), time=datetime.now(timezone.utc).isoformat()))
        raise
