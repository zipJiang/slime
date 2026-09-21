"""Prepare the complete deterministic LocBench schedule from a validated critic."""
import argparse
import json
from pathlib import Path
from runtime_active import EXPERIMENT,MODEL,digest
from ppo_protocol import candidate,schedule


def prepare(path,output,updates=120,batch_size=4):
    value=candidate(path)
    split=json.loads((EXPERIMENT/'data/split.json').read_text())
    critic_recipe=json.loads((Path(path).parent/'recipe.json').read_text())
    collection=Path(critic_recipe['arguments']['loc_collection'])
    manifest=json.loads((collection/'manifest.json').read_text())
    plan=schedule(split,updates=updates,batch_size=batch_size,warmup_ids=manifest['train_ids'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit=output/'schedule-audit.json'
    if audit.exists() and json.loads(audit.read_text())!=plan:raise ValueError('Existing PPO schedule differs')
    audit.write_text(json.dumps(plan,indent=2)+'\n')
    prompts=output/'schedule.jsonl'
    prompts.write_text(''.join(json.dumps(dict(prompt=[dict(role='user',content=q)],metadata=dict(query_id=q)))+'\n' for q in plan['questions']))
    model=value['actor_identity']['model'] if value.get('actor_identity') else MODEL
    return dict(model=model,schedule=str(prompts.resolve()),audit=str(audit.resolve()),
        schedule_sha256=digest(audit),candidate_sha256=digest(path))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=120);p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args();print(json.dumps(prepare(args.candidate,args.output,args.updates,args.batch_size)),flush=True)
