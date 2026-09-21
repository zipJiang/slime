"""Keep known lease deadlines usable during Slurm controller outages."""
import datetime
import subprocess
import time


def refresh(jobs, previous, *, now=None, query=subprocess.check_output):
    now = time.time() if now is None else now
    jobs = list(jobs)
    cached = {row['job']: row for row in previous}
    errors = {}
    try:
        output = query(['squeue', '--jobs='+','.join(jobs), '--noheader', '--format=%i|%e'],
                       text=True, stderr=subprocess.STDOUT, timeout=10)
    except (subprocess.SubprocessError, OSError) as exc:
        output = ''
        errors['query'] = str(exc)
    rows = {}
    for line in output.splitlines():
        fields = line.strip().split('|')
        if len(fields) == 2 and fields[0] in jobs:
            rows[fields[0]] = fields[1]
    leases = []
    for job in jobs:
        end = rows.get(job)
        try:
            expiry = datetime.datetime.fromisoformat(end).timestamp()
            source = 'slurm'
        except (ValueError, TypeError, OverflowError):
            errors[job] = 'Missing or invalid lease deadline; retaining last known deadline'
            end = cached.get(job, {}).get('end')
            expiry = datetime.datetime.fromisoformat(end).timestamp() if end else now
            source = 'cached' if end else 'unknown'
        leases.append(dict(job=job, end=end, remaining_seconds=expiry-now, source=source))
    # Unknown deadlines request a graceful checkpoint, never an immediate kill.
    return leases, errors
