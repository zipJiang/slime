"""Diagnostic-only replay of the failed round; never permits an optimizer step."""
import gzip
import json
from pathlib import Path
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample
SOURCE=Path('/weka/projects/bvandur1/zjiang31/locbench-vine-9b/runs/locbench-vine-g4-c12-sep18-v3/rollouts/train-0030')
def generate_rollout(args, rollout_id, data_source, evaluation=False):
    assert rollout_id == 30 and not evaluation
    assert args.rollout_data_postprocess_path == 'diagnostic_audit_sep18.check'
    run=Path(args.save).parent
    freeze=json.loads((run/'collection-freeze.json').read_text())
    originals=data_source.get_samples(args.rollout_batch_size)
    cases=[group[0].metadata['query_id'] for group in originals]
    assert cases == json.loads((SOURCE/'questions.json').read_text())
    directory=run/'rollouts'/f'train-{rollout_id:04d}'
    directory.mkdir(parents=True,exist_ok=False)
    old=json.loads((SOURCE/'contract.json').read_text())
    for key in ('recipe_id','estimator','policy_version','server_weight_version'):
        assert old[key] == freeze[key], (key,old[key],freeze[key])
    (directory/'contract.json').write_text(json.dumps(old,indent=2))
    (directory/'diagnostic-replay.json').write_text(json.dumps(dict(source=str(SOURCE),optimizer_steps_allowed=0),indent=2))
    samples=[]
    for group,case in enumerate(cases):
        with gzip.open(SOURCE/f'group-{group:03d}.actor.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line)
                sample=Sample(index=len(samples),rollout_id=group,group_index=group,
                    prompt=case,tokens=row['tokens'],response_length=row['response_length'],
                    loss_mask=row['loss_mask'],rollout_log_probs=row['rollout_log_probs'],
                    reward=row['reward'],metadata=dict(prepared_record=row),status=Sample.Status.COMPLETED)
                sample.weight_versions=[freeze['server_weight_version']]
                samples.append(sample)
    return RolloutFnTrainOutput(samples=samples,metrics={'diagnostic_only':1})
