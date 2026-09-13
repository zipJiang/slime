"""Promote an offline critic candidate only after a real zero-warmup PPO pilot."""
import json
from pathlib import Path

from warmstart_candidate import digest, require_pilot_candidate


def read(path):
    return json.loads(Path(path).read_text())


def promote(candidate_path, pilot_path):
    candidate_path=Path(candidate_path).resolve()
    pilot_path=Path(pilot_path).resolve()
    candidate=require_pilot_candidate(candidate_path)
    pilot=read(pilot_path)
    if pilot.get('schema')!='browsecomp-zero-warmup-pilot-v1':
        raise ValueError('Unknown zero-warmup pilot schema')
    if pilot.get('candidate_sha256')!=digest(candidate_path):
        raise ValueError('Pilot used a different critic candidate')
    if (pilot.get('context_function_sha256')!=candidate['context_function_sha256']
            or pilot.get('num_critic_only_steps')!=0 or pilot.get('start_rollout_id')!=0):
        raise ValueError('Pilot conditioning or zero-warmup cursor contract changed')
    updates=pilot.get('completed_joint_updates')
    if not isinstance(updates,int) or updates<2:
        raise ValueError('Pilot must complete at least two joint actor/critic updates')
    initialization=pilot['initialization']
    if Path(initialization['actor']['load']).resolve()!=Path(candidate['base_actor']).resolve():
        raise ValueError('Pilot actor did not start from the candidate base model')
    expected_critic=candidate['critic']
    if (Path(initialization['critic']['load']).resolve()!=Path(expected_critic['load']).resolve()
            or initialization['critic']['ckpt_step']!=expected_critic['ckpt_step']):
        raise ValueError('Pilot critic did not start from the candidate checkpoint')
    for role in ('actor','critic'):
        state=initialization[role]
        if state['cursors']!=[0,0,0,0] or not all(item['fresh'] for item in state['optimizers']):
            raise ValueError(f'Pilot {role} did not start at cursor zero with a fresh optimizer')
    batches=pilot.get('batches',[])
    if (len(batches)!=updates or any(not row.get(key) for row in batches
            for key in ('target_replay_passed','on_policy_passed','context_contract_passed'))
            or any(row.get('infrastructure_failures',0) for row in batches)):
        raise ValueError('Pilot batches lack complete target, policy, context, or infrastructure audits')
    terminals=successes=0
    for row in batches:
        terminal_count=row.get('terminals');success_count=row.get('successes')
        if (type(terminal_count) is not int or terminal_count<=0
                or isinstance(success_count,bool) or not isinstance(success_count,(int,float))
                or not float(success_count).is_integer()
                or not 0<=success_count<=terminal_count):
            raise ValueError('Pilot batch has invalid terminal outcome evidence')
        terminals+=terminal_count;successes+=int(success_count)
    if successes==0 or successes==terminals:
        raise ValueError('Pilot must observe both successful and failed terminal branches')
    publications=pilot.get('critic_publications',[])
    if (len(publications)<updates+1 or any(not row.get('passed') or not row.get('deterministic')
            or row['max_abs_error']>row['tolerance'] for row in publications)):
        raise ValueError('Pilot critic publications lack native/portable equivalence')
    checkpoint=pilot['checkpoint']
    for role in ('actor','critic'):
        audit=checkpoint[role]
        if (not audit['full_storage_read'] or not audit['finite_tensors']
                or audit['optimizer_steps']!=[updates]):
            raise ValueError(f'Pilot {role} checkpoint lacks complete readback')
    if pilot.get('scientific_rejection') is not False:
        raise ValueError('Scientifically rejected pilot cannot authorize a long run')
    return dict(schema='browsecomp-critic-long-run-warmstart-v1',ready_for_long_run=True,
        candidate=str(candidate_path),candidate_sha256=digest(candidate_path),
        pilot=str(pilot_path),pilot_sha256=digest(pilot_path),joint_updates=updates,
        actor_base=candidate['base_actor'],critic=candidate['critic'],
        portable_inference=candidate['portable_inference'],
        context_function_sha256=candidate['context_function_sha256'])


def require_long_run(path):
    manifest=read(path)
    if manifest.get('schema')!='browsecomp-critic-long-run-warmstart-v1' or not manifest.get('ready_for_long_run'):
        raise ValueError('Critic has not passed the zero-warmup long-run gate')
    for name in ('candidate','pilot'):
        if digest(manifest[name])!=manifest[f'{name}_sha256']:
            raise ValueError(f'Promoted critic {name} evidence changed')
    expected=promote(manifest['candidate'],manifest['pilot'])
    if manifest!=expected:
        raise ValueError('Long-run manifest does not exactly reproduce the pilot gate')
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('candidate',type=Path)
    parser.add_argument('pilot',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=promote(args.candidate,args.pilot)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
