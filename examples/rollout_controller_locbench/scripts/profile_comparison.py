"""Run a fresh paired pilot against restored, immutable actor weights; no optimizer steps."""
import json
import os
from pathlib import Path
import subprocess
from runtime_v2 import EXPERIMENT, HARNESS, digest
from ppo_collection import PY
from rollout_version import engine_version
from train_critic import write


def compare(args,run,manager,behavior_version,start):
    if not args.loc_profile_comparison_only or not args.loc_resume_run:
        raise ValueError('Profile pilot is isolated and restored only')
    write(run/'status.json',dict(stage='comparing-compaction-profiles',updates_will_be_discarded=True))
    output=run/'profile-pilot'
    cmd=[PY,str(EXPERIMENT/'scripts/profile_pilot.py'),'--output',str(output),
        '--url',f'http://{args.sglang_router_ip}:{args.sglang_router_port}',
        '--policy-version',f'actor-{start:04d}','--server-weight-version',behavior_version]
    with (run/'profile-pilot.log').open('x') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=HARNESS,
            env=dict(os.environ,PYTHONPATH=str(HARNESS)))
    if engine_version(manager,expected_engines=args.rollout_num_gpus)!=behavior_version:
        raise ValueError('Pilot actor weights changed')
    result=json.loads((output/'comparison.json').read_text())
    write(run/'profile-comparison-complete.json',dict(completed=True,screen=result,
        comparison_sha256=digest(output/'comparison.json'),optimizer_steps=0,
        next='Review repetition examples and native memory qualification before any production promotion'))
