import json
import pytest
from synthetic_runtime import LimitedRead,make_world,environment
from runtime_v2 import make_world as old_world,MODEL
from examples.locbench.dataset import Case
from examples.locbench.metrics import Gold
from step_controller.generation import Policy,PolicyFormat
from step_controller.harness import ToolError

class Repo:
 def read(self,path,start,lines):return dict(path=path,start=start,lines=lines,next_start=start+lines)

@pytest.mark.asyncio
async def test_read_contract_rejects_oversized_requests_and_preserves_paging():
 tool=LimitedRead(Repo())
 assert tool.parameters['properties']['lines']==dict(type='integer',minimum=1,maximum=60,default=60)
 assert json.loads(await tool.call({'path':'file.py'}))['next_start']==61
 assert json.loads(await tool.call({'path':'file.py','start':61,'lines':'60'}))['next_start']==121
 for lines in (0,61,120,True):
  with pytest.raises(ToolError):await tool.call({'path':'file.py','lines':lines})


def test_world_exposes_same_read_limit_as_dispatch_and_keeps_old_world_unchanged():
 from transformers import AutoTokenizer
 fmt=PolicyFormat.resolve(MODEL,tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True),profile='qwen_xml')
 from examples.locbench.dataset import load_cases
 from runtime_v2 import EXPERIMENT
 case=load_cases(EXPERIMENT/'data/train.jsonl')[0];policy=Policy(format=fmt)
 repo=Repo();repo.commit=case.base_commit
 old=old_world(case,repo,policy)[0]
 runner,_,_,schemas=make_world(case,repo,policy)
 assert next(s for s in schemas if s['function']['name']=='read')['function']['parameters']['properties']['lines']['maximum']==60
 assert runner._env._by_name['read'].parameters['properties']['lines']['maximum']==60
 assert old._env._by_name['read'].parameters['properties']['lines']['maximum']==120
 from types import SimpleNamespace
 state=SimpleNamespace(calls_remaining=20,calls_max=20)
 assert not runner._compactor.trigger.fires(range(24000),state)
 assert runner._compactor.trigger.fires(range(24577),state)
 assert not old._compactor.trigger.fires(range(24577),state)
 assert environment()['fold_reply_limit']==16384
 assert make_world(case,repo,policy,compact=False)[0]._compactor is None
