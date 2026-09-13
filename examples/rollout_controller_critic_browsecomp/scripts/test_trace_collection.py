"""Exercise multi-endpoint collection and exact saved supervision without network."""
import os
from pathlib import Path
import subprocess
import sys


def test_fresh_trace_collection_and_readback():
    experiment = Path(__file__).resolve().parents[1]
    code = r'''
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import httpx
import trace_collect as c
import trace_audit_collection as a
from transformers import AutoTokenizer
from step_controller.generation import PolicyFormat, SamplingParams
from step_controller.generation.slime import SlimePolicy

tokenizer=AutoTokenizer.from_pretrained('/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a',local_files_only=True)
used=set()
def generator(**kwargs):
    url=kwargs['base_url']
    async def post(endpoint,payload):
        used.add(url)
        reply='<tool_call>\n<function=submit>\n<parameter=answer>London</parameter>\n</function>\n</tool_call>'
        tokens=tokenizer.encode(reply,add_special_tokens=False)
        return dict(text=reply,meta_info=dict(output_token_logprobs=[[-.1,t,None] for t in tokens],finish_reason={'type':'stop'},weight_version='1'))
    return SlimePolicy(post,url+'/generate',model='Qwen/Qwen3.5-9B',format=kwargs['format'],version='base',default_params=kwargs['default_params'])
class Client:
    def __init__(self,*args,**kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self,*args): pass
class HTTP(Client):
    count=0
    async def post(self,url,**kwargs):
        correct=self.count in (0,2)
        self.count+=1
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(choices=[dict(message=dict(content='EQUIVALENT' if correct else 'DIFFERENT'),finish_reason='stop')]))
c.FoldGenerator=generator
c.client_class=lambda path:Client
httpx.AsyncClient=HTTP
with tempfile.TemporaryDirectory() as directory:
    args=SimpleNamespace(model=tokenizer.name_or_path,cases=c.EXPERIMENT.parents[2]/'rollout-controller/data/browsercomp-plus/cases.private.jsonl',output=Path(directory),actor_url=[f'http://actor-{i}' for i in range(5)],judge_url='http://judge',retriever_url='http://retriever',retriever_code=Path('.'),concurrency=2,pilot=True)
    asyncio.run(c.main(args))
    report=a.audit(Path(directory))
    assert report['traces']==4 and report['checkpoints']==4 and report['exact_snapshot_context_readback']
    assert len(used)>=2
    manifest=json.loads((Path(directory)/'manifest.json').read_text())
    assert (manifest['task_limit'],manifest['max_context'],manifest['actor_max_tokens'])==(96,98304,16384)
    # Tampering with the exact saved supervision must be rejected.
    path=next(Path(directory).glob('*/*/sample-*.json'))
    row=json.loads(path.read_text());row['collection_manifest_sha256']='0'*64;path.write_text(json.dumps(row))
    try: a.audit(Path(directory))
    except ValueError as exc: assert 'manifest' in str(exc)
    else: raise AssertionError('Tampered trace was accepted')
print('Fresh TRACE collection, multi-endpoint routing, exact context replay and tamper rejection passed')
'''
    env = dict(os.environ, BROWSECOMP_PROFILE='trace96k', PYTHONPATH=os.pathsep.join([
        str(experiment / 'snapshots/harness-trace-v1'), str(experiment / 'scripts')]))
    result = subprocess.run([sys.executable, '-c', code], cwd=experiment / 'snapshots/harness-trace-v1',
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
