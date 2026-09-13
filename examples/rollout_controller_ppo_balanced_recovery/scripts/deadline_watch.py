"""Request a drained save before a known allocation deadline; never kill workers."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main(run, deadline):
    while True:
        recipe_file = run/'recipe.json'
        if recipe_file.exists():
            recipe = json.loads(recipe_file.read_text())
            driver = recipe['driver_step']
            query = subprocess.run(['squeue', '--steps='+driver, '-h', '-o', '%i'],
                                   capture_output=True, text=True)
            if query.returncode == 0 and driver not in query.stdout.split():
                print('Driver exited; deadline watcher finished.', flush=True)
                return
        if any((run/name).exists() for name in ('completed.json', 'paused.json', 'failed.json')):
            return
        if time.time() >= deadline:
            run.mkdir(parents=True, exist_ok=True)
            (run/'STOP').touch(exist_ok=True)
            (run/'deadline-stop.json').write_text(json.dumps(dict(
                requested_unix=time.time(), deadline_unix=deadline,
                action='drain_and_save_at_batch_boundary'), indent=2)+'\n')
            print('Requested drained checkpoint before allocation expiry.', flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    main(args.run,args.deadline)
