"""Audit committed native actor checkpoints while the owning driver is live."""
import json
from pathlib import Path
import subprocess
import sys
import time
E = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]).resolve()
while not (run/'recipe.json').exists():
    time.sleep(5)
recipe = json.loads((run/'recipe.json').read_text())
step = recipe['driver_step']
while True:
    tracker = run/'actor/latest_checkpointed_iteration.txt'
    latest = int(tracker.read_text()) if tracker.exists() else -1
    for checkpoint in sorted((run/'actor').glob('iter_*')):
        if not checkpoint.is_dir() or not checkpoint.name[5:].isdigit():
            continue
        iteration = int(checkpoint.name[5:])
        if iteration > latest:
            continue
        report = checkpoint.with_name(checkpoint.name+'-readback.json')
        if report.exists():
            evidence = json.loads(report.read_text())
            if (evidence.get('full_storage_read') and evidence.get('finite_tensors')
                    and evidence.get('optimizer_steps') == [iteration+1]):
                continue
        logdir = run/'validation-logs'
        logdir.mkdir(exist_ok=True)
        with (logdir/(checkpoint.name+'-actor.log')).open('a') as log:
            result = subprocess.run(['bash',str(E/'scripts/sif.sh'),'python',
                str(E/'scripts/audit_checkpoint.py'),str(checkpoint),
                '--role','actor','--expected-steps',str(iteration+1)],
                stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (run/'STOP').touch()
            raise RuntimeError('Checkpoint audit failed: '+str(checkpoint))
        print('Verified', checkpoint, flush=True)
    try:
        probe = subprocess.run(['squeue','--steps='+step,'-h','-o','%i'],
            capture_output=True,text=True,timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        # This process runs beside a large Ray/SGLang topology.  A transient
        # local fork failure or Slurm timeout says nothing about checkpoint
        # validity or driver liveness, so leave training alone and re-observe.
        print(f'Transient driver-step observation failure: {exc!r}',
              file=sys.stderr, flush=True)
        time.sleep(30)
        continue
    if probe.returncode == 0 and step not in probe.stdout.split():
        # A successful observation of a missing step is authoritative termination.
        break
    time.sleep(30)
