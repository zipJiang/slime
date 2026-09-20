"""Retry controller connection failures, never a started remote workload."""
import collections
from pathlib import Path
import subprocess
import sys
import time

MARKER = "__SLURM_REMOTE_COMMAND_STARTED__"
DELAYS = (5, 10, 20, 30, 30)

def wrap(command):
    return [sys.executable, str(Path(__file__).resolve()), *command]

def retryable(code, errors, started):
    text = errors.lower()
    return (code != 0 and not started and
            "unable to confirm allocation" in text and
            "unable to contact slurm controller" in text)

def main():
    command = sys.argv[1:]
    assert command[0] == "srun"
    # All generated srun options use --key=value; the first non-option is remote.
    boundary = next(i for i in range(1, len(command)) if not command[i].startswith("--"))
    command[boundary:boundary] = ["bash", "-c", 'echo '+MARKER+' >&2; exec "$@"', "slurm-retry"]
    for attempt in range(len(DELAYS) + 1):
        errors = collections.deque(maxlen=100)
        started = False
        child = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
        for line in child.stderr:
            if MARKER in line:
                started = True
            else:
                sys.stderr.write(line); sys.stderr.flush()
                errors.append(line)
        code = child.wait()
        if not retryable(code, "".join(errors), started) or attempt == len(DELAYS):
            return code if code >= 0 else 128-code
        delay = DELAYS[attempt]
        print(f"Slurm controller unavailable before launch; retry {attempt+1}/{len(DELAYS)} in {delay}s", file=sys.stderr, flush=True)
        time.sleep(delay)

if __name__ == "__main__":
    sys.exit(main())
