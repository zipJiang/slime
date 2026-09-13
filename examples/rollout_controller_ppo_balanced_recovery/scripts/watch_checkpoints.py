"""Validate both saved roles and the exported actor without stopping training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def validation_command(command, driver_step):
    """Run the pinned SIF on the driver's allocation if this host lacks it."""
    if shutil.which('apptainer'):
        return command
    job = str(driver_step).partition('.')[0]
    if not job.isdecimal():
        raise ValueError('Checkpoint validation requires a numeric driver allocation')
    return ['srun', f'--jobid={job}', '--overlap', '--mem=0', '--cpu-bind=none',
            '-N1', '-n1', '-c4', *command]


def require_success(outcomes):
    failed = [name for name, report in outcomes.items() if not report['passed']]
    if failed:
        raise RuntimeError(f'Checkpoint validation failed: {failed}')


def watch(run, once=False, retry_failed=False):
    # The SIF wrapper changes cwd; subprocess inputs must be absolute.
    run = run.resolve()
    experiment = Path(__file__).resolve().parents[1]
    scripts = experiment/'scripts'
    recipe = json.loads((run/'recipe.json').read_text())
    args = recipe['arguments']
    warmup = args['num_critic_only_steps']
    evidence = run/'checkpoint-watch.json'
    outcomes = json.loads(evidence.read_text()) if evidence.exists() else {}
    logs = run/'validation-logs'
    logs.mkdir(exist_ok=True)
    while True:
        for actor in sorted((run/'actor').glob('iter_*')):
            match = re.fullmatch(r'iter_(\d+)', actor.name)
            prior = outcomes.get(actor.name)
            if not match or (prior and (prior['passed'] or not retry_failed)):
                continue
            iteration = int(match[1])
            actor_updates = max(0, iteration+1-warmup)
            critic = run/'critic'/actor.name
            hf = run/'hf'/actor.name
            # The driver saves both roles synchronously. Require both completed
            # tracker files; merely seeing a directory is not a completed save.
            trackers = [run/role/'latest_checkpointed_iteration.txt' for role in ('actor','critic')]
            if not all(p.exists() and int(p.read_text().strip()) >= iteration for p in trackers):
                continue
            if not all((p/'.metadata').exists() and (p/'common.pt').exists() for p in (actor,critic)):
                continue
            if not (hf/'model.safetensors.index.json').exists():
                continue
            commands = []
            for role, path, expected in (('actor', actor, actor_updates), ('critic', critic, iteration+1)):
                commands.append((role, ['bash', str(scripts/'sif.sh'), 'python', str(scripts/'audit_checkpoint.py'),
                    str(path), '--role', role, '--expected-steps', str(expected)]))
            export = ['bash', str(scripts/'sif.sh'), 'python', str(scripts/'finalize_hf.py'),
                      '--source', args['hf_checkpoint'], '--target', str(hf)]
            if actor_updates == 0:
                export.append('--expect-unchanged')
            commands.append(('export', export))
            outcome = dict(started_at=datetime.now(timezone.utc).isoformat(),
                actor_updates=actor_updates, critic_updates=iteration+1, checks={})
            history = [] if prior is None else [*prior.get('previous_attempts', []),
                {k:v for k,v in prior.items() if k != 'previous_attempts'}]
            if history:
                outcome['previous_attempts'] = history
            for kind, command in commands:
                path = logs/f'{actor.name}-attempt{len(history)+1}-{kind}.log'
                with path.open('w') as stream:
                    result = subprocess.run(validation_command(command, recipe['driver_step']),
                        stdout=stream, stderr=subprocess.STDOUT)
                outcome['checks'][kind] = dict(exit_code=result.returncode, log=str(path.resolve()))
            outcome['passed'] = all(v['exit_code'] == 0 for v in outcome['checks'].values())
            outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
            outcomes[actor.name] = outcome
            write_json(evidence, outcomes)
            print(json.dumps(dict(checkpoint=actor.name, **outcome)), flush=True)
        if once:
            require_success(outcomes)
            return
        probe = subprocess.run(['squeue', '--steps='+recipe['driver_step'], '-h', '-o', '%i'],
                               text=True, capture_output=True)
        if probe.returncode == 0 and recipe['driver_step'] not in probe.stdout.split():
            require_success(outcomes)
            print('Driver is terminal; checkpoint observation finished.', flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    watch(args.run, args.once, args.retry_failed)
