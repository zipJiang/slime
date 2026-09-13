"""Replay a settled BrowserComp pilot batch and verify every training target."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle

from collect import context as serialize_context
from examples.browsercomp_plus.env import RetrievalArchive, SearchEnv, load_cases
from judge_contract import verdict
from step_controller import DirectBranchTdEstimator
from step_controller.export import to_samples
from step_controller.preparation import prepare_samples
from step_controller.reward.config import RewardConfig
from targets import split_targets
from provenance import function_sha256
from semantic_judge import contract as judge_contract


RECIPE_ID='browsecomp-zero-warmup-pilot-v1'
EXPERIMENT=Path(__file__).resolve().parents[1]
ROOT=EXPERIMENT.parents[2]
HARNESS=EXPERIMENT/'snapshots/harness'
CASES=ROOT/'rollout-controller/data/browsercomp-plus/cases.private.jsonl'


def rows(path):
    with gzip.open(path,'rt') as stream: return [json.loads(line) for line in stream]


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def audit_judge(records, *, question, references, terminal_pairs):
    question_hash=hashlib.sha256(question.encode()).hexdigest()
    reference_hash=hashlib.sha256(json.dumps(tuple(str(r) for r in references)).encode()).hexdigest()
    observed=[]
    for record in records:
        if (record.get('version')!=judge_contract()['version']
                or record.get('model')!=judge_contract()['model']
                or record.get('question_sha256')!=question_hash
                or record.get('references_sha256')!=reference_hash
                or not isinstance(record.get('attempt'),int) or not 1<=record['attempt']<=3):
            raise ValueError('Judge provenance differs from the frozen semantic contract')
        parsed=verdict(record.get('response',''),record.get('finish_reason'))
        if type(record.get('correct')) is not bool or record['correct']!=parsed:
            raise ValueError('Stored semantic label differs from strict verdict parsing')
        submitted=record.get('submitted')
        if not isinstance(submitted,str) or not submitted.strip():
            raise ValueError('Judge evidence contains an empty submission')
        observed.append((submitted,float(parsed)))
    if Counter(observed)!=Counter(terminal_pairs):
        raise ValueError('Terminal outcomes differ from semantic judge evidence')


def audit(directory):
    from transformers import AutoTokenizer
    directory=Path(directory)
    contract=json.loads((directory/'contract.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if contract['recipe_id']!=RECIPE_ID or contract['estimator']!='direct_branch_td':
        raise ValueError('Unknown pilot preparation contract')
    if hashlib.sha256(Path(__file__).with_name('pilot_collect.py').read_bytes()).hexdigest()!=contract['collector_sha256']:
        raise ValueError('Pilot collector source changed after collection')
    if (contract.get('checkpoint_context_function_sha256')!=function_sha256(
            EXPERIMENT/'scripts/collect.py','context')
            or contract.get('harness_manifest_sha256')!=hashlib.sha256(
                (HARNESS/'source-manifest.json').read_bytes()).hexdigest()
            or contract.get('split_sha256')!=hashlib.sha256(
                (EXPERIMENT/'data/split.json').read_bytes()).hexdigest()
            or contract.get('cases_sha256')!=hashlib.sha256(CASES.read_bytes()).hexdigest()
            or contract.get('judge')!=judge_contract()):
        raise ValueError('Pilot source, data, context, or judge contract changed after collection')
    infrastructure=Path(contract.get('infrastructure_manifest',''))
    if (not infrastructure.is_file()
            or hashlib.sha256(infrastructure.read_bytes()).hexdigest()!=
                contract.get('infrastructure_manifest_sha256')
            or json.loads(infrastructure.read_text()).get('schema')!=
                'browsecomp-zero-warmup-pilot-infrastructure-v1'):
        raise ValueError('Pilot infrastructure manifest changed after collection')
    if (summary.get('recipe_id')!=RECIPE_ID
            or any(summary.get(key)!=contract[key] for key in
                   ('policy_version','server_weight_version','value_version'))):
        raise ValueError('Pilot summary version lineage differs from its contract')
    tokenizer=AutoTokenizer.from_pretrained(contract['checkpoint'],local_files_only=True)
    tools=SearchEnv(RetrievalArchive()).schemas
    cases=load_cases(CASES)
    rc=RewardConfig(value_version=contract['value_version'],
        value_prior_strength=contract['prior_strength'],
        config_id=f"browsecomp-ppo-kappa-{contract['prior_strength']:g}")
    reports=[]
    for group,question in enumerate(contract['case_ids']):
        stem=directory/f'group-{group:03d}'
        with gzip.open(stem.with_suffix('.native.pkl.gz'),'rb') as stream: state=pickle.load(stream)
        prepared=prepare_samples(state,estimator=DirectBranchTdEstimator(),
            reward_config=rc,behavior_version=contract['policy_version'])
        actor,critic=split_targets(to_samples(prepared,group_index=group))
        for expected,lane in ((actor,'actor'),(critic,'critic')):
            actual=rows(stem.with_suffix(f'.{lane}.jsonl.gz'))
            if len(actual)!=len(expected): raise ValueError(f'{question}: {lane} record count')
            for a,b in zip(actual,expected,strict=True):
                a=dict(a,metadata=dict(a['metadata']))
                for key,value in dict(query_id=question,policy_version=contract['policy_version'],
                        server_weight_version=contract['server_weight_version']).items():
                    if a['metadata'].pop(key)!=value: raise ValueError(f'{question}: {key}')
                if lane=='critic':
                    if a['metadata'].pop('target_source')!='direct_branch_mean':
                        raise ValueError(f'{question}: critic target source')
                    boundary=a['metadata'].pop('terminal_boundary')
                    node=state.nodes[b['metadata']['node_id']]
                    if type(boundary) is not bool or boundary!=node.payload.done:
                        raise ValueError(f'{question}: terminal boundary')
                    if a['context']!=serialize_context(node.payload,tools,tokenizer):
                        raise ValueError(f'{question}: critic conditioning context changed')
                    diagnostics=a['metadata'].get('diagnostics',{})
                    if boundary:
                        if float(a['target'])!=0 or float(diagnostics.get('mean_return',1))!=0:
                            raise ValueError(f'{question}: terminal critic target is not fixed zero')
                    elif (int(diagnostics.get('direct_branches',0))<=0
                            or float(a['target'])!=float(diagnostics.get('mean_return','nan'))
                            or not 0<=float(a['target'])<=1):
                        raise ValueError(f'{question}: invalid direct-branch critic supervision')
                if a!=b: raise ValueError(f'{question}: {lane} replay differs')
        turns={id(turn):turn for node in state.nodes.values() for turn in node.payload.turns}
        cost=dict(input_tokens=sum(len(t.prefix) for t in turns.values() if t.tokens),
            output_tokens=sum(len(t.tokens) for t in turns.values()),
            generations=sum(bool(t.tokens) for t in turns.values()))
        result=json.loads(stem.with_suffix('.json').read_text())
        if cost!=result['cost'] or cost['output_tokens']!=sum(sum(r['loss_mask']) for r in actor):
            raise ValueError(f'{question}: generated token accounting mismatch')
        if len(result['passes'])!=2 or any(state.stats.get(k) for k in ('failures','score_failures')):
            raise ValueError(f'{question}: incomplete or failed search')
        if result['terminal_correct']!=sum(n.payload.reward_outcome for n in state.nodes.values() if n.payload.done):
            raise ValueError(f'{question}: semantic outcome evidence mismatch')
        judge=json.loads(stem.with_suffix('.judge.json').read_text())['records']
        terminal_pairs=Counter((str(n.payload.state.answer),float(n.payload.reward_outcome))
            for n in state.nodes.values() if n.payload.done and
            n.payload.state.answer is not None and str(n.payload.state.answer).strip())
        empty_terminals=[n for n in state.nodes.values() if n.payload.done and
            (n.payload.state.answer is None or not str(n.payload.state.answer).strip())]
        if len(judge)!=result['judge_calls'] or any(float(n.payload.reward_outcome)!=0 for n in empty_terminals):
            raise ValueError(f'{question}: judge evidence mismatch')
        audit_judge(judge,question=cases[question].question,
            references=cases[question].answers,terminal_pairs=terminal_pairs)
        tags=Counter()
        for turn in turns.values(): tags[turn.tag]+=len(turn.tokens)
        reports.append(dict(query_id=question,actor_spans=len(actor),critic_checkpoints=len(critic),
            cost=cost,generated_tokens_by_tag=dict(tags),judge_calls=len(judge),
            terminal_count=result['terminal_count'],terminal_correct=result['terminal_correct']))
    if summary['cost']!={key:sum(r['cost'][key] for r in reports)
            for key in ('input_tokens','output_tokens','generations')}:
        raise ValueError('Whole-batch cost summary differs from replay')
    report=dict(passed=True,batch=str(directory.resolve()),groups=reports,
        policy_version=contract['policy_version'],value_version=contract['value_version'],
        recipe_id=RECIPE_ID,estimator='direct_branch_td',context_contract_passed=True,
        infrastructure_failures=0)
    write(directory/'target-replay-audit.json',report)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path);args=parser.parse_args()
    print(json.dumps(audit(args.batch),indent=2))
