"""Read-only target replay checks as this run produces complete artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from audit_batch import audit


def main(run):
    while True:
        for directory in sorted((run/'rollouts').glob('train-*')):
            if not (directory/'contract.json').exists():
                continue
            whole = (directory/'summary.json').exists()
            if whole:
                pending = [] if (directory/'target-replay-audit.json').exists() else [None]
            else:
                pending = [int(m[1]) for p in sorted(directory.glob('group-*.json'))
                    if (m := re.fullmatch(r'group-(\d+)\.json', p.name))
                    and not (directory/f'group-{int(m[1]):03d}.target-replay-audit.json').exists()]
            for group in pending:
                try:
                    report = audit(directory, group)
                    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                        batch=directory.name, group=group, passed=True,
                        actor_spans=sum(r['actor_spans'] for r in report['groups']),
                        critic_checkpoints=sum(r['critic_checkpoints'] for r in report['groups']))), flush=True)
                except Exception as exc:
                    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                        batch=directory.name, group=group, passed=False, error=repr(exc))), flush=True)
                    raise
        if any((run/name).exists() for name in ('completed.json', 'paused.json', 'failed.json')):
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
