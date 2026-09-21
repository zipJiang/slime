"""Explicit 24K/8K/60-line candidate; preserves the original frozen PPO runtime."""
from runtime_v2 import contract,digest,make_world as original_world
from examples.locbench.env import RepositoryTool
from step_controller.harness import ToolError
from step_controller.harness.compaction.triggers import AnyOf,PromptTokens,StateCounter
READ_LIMIT=60

def environment():
 result=contract();result.update(profile='locbench-compact24-reply8-read60-v1',prompt_limit=24576,actor_reply_limit=8192,read_line_limit=READ_LIMIT,runtime_sha256=digest(__file__))
 return result

class LimitedRead(RepositoryTool):
 def __init__(self,repo):
  super().__init__(repo,'read')
  self.description=self.description.replace('120 lines',f'{READ_LIMIT} lines')+' Continue with next_start to read another page.'
  self.parameters['properties']['lines'].update(maximum=READ_LIMIT,default=READ_LIMIT)
 async def call(self,params,**kwargs):
  args=self.verify_args(params);lines=args.get('lines',READ_LIMIT)
  if type(lines) is not int or not 1<=lines<=READ_LIMIT:raise ToolError(f'lines must be between 1 and {READ_LIMIT}; use next_start to continue')
  return await super().call(dict(args,lines=lines),**kwargs)

def make_world(case,repo,policy,*,compact=True,read_limit=READ_LIMIT):
 runner,workspace,prompt,_=original_world(case,repo,policy)
 if read_limit not in (READ_LIMIT,120):raise ValueError('Unsupported diagnostic read limit')
 if read_limit==READ_LIMIT:
  env=runner._env;tool=LimitedRead(repo);env._tools=[tool if t.name=='read' else t for t in env.tools];env._by_name['read']=tool;runner._tools=env.schemas
 if compact:runner._compactor.trigger=AnyOf(PromptTokens(24576),StateCounter())
 else:runner._compactor=None
 return runner,workspace,prompt,runner._tools
