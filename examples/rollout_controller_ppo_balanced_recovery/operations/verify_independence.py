"""Run on the supervisor host to verify all launchers belong to its batch cgroup."""
import argparse
import json
import os
from pathlib import Path
import time

parser=argparse.ArgumentParser()
parser.add_argument('state',type=Path)
args=parser.parse_args()
p=args.state
ready=json.loads((p/'ready.json').read_text())
assert os.uname().nodename==ready['host'], 'Run this audit on the supervisor host'
records=json.loads((p/'processes.json').read_text())
expected=set(ready.get('ray_labels', ['ray-gh202','ray-gh203','ray-gh121','ray-gh130'])) | {
    'train','target-audits','checkpoints','evaluation','deadline'}
assert records.keys()==expected, records.keys()
rows={}
for label,record in records.items():
    pid=record['pid']
    os.kill(pid,0)
    group=Path(f'/proc/{pid}/cgroup').read_text().strip()
    assert f'job_{ready["job"]}/step_batch/' in group,(label,group)
    fds={str(fd):str(Path(f'/proc/{pid}/fd/{fd}').readlink()) for fd in range(3)}
    assert fds['0']=='/dev/null',(label,fds)
    assert fds['1']==record['log'] and fds['2']==record['log'],(label,fds)
    assert os.getsid(pid)==pid,(label,pid)
    rows[label]=dict(pid=pid,cgroup=group,stdin=fds['0'],output=fds['1'],session=pid)
report=dict(passed=True,checked_unix=time.time(),host=ready['host'],supervisor_job=ready['job'],processes=rows)
(p/'independence-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,host=ready['host'],supervisor_job=ready['job'],checked=list(rows))))
