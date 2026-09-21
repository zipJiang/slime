from types import SimpleNamespace
from synthetic_targets import export
from step_controller.harness import Turn

def test_rejected_request_stays_in_conditioning_but_not_loss():
 bad=SimpleNamespace(messages=({'role':'tool','content':'error: lines must be between 1 and 60; use next_start to continue'},))
 turns=(Turn(prefix=(1,2),tokens=(3,),tag='fold'),Turn(prefix=(6,7),tokens=(8,9),transition=bad),Turn(prefix=(6,7,8,9,10),tokens=(11,12)))
 rows=export(turns,'example-id','native-source',0)
 assert [t for r in rows for t,m in zip(r['tokens'],r['loss_mask']) if m]==[3,11,12]
 task=next(r for r in rows if r['metadata']['tag']=='task')
 assert task['tokens']==[6,7,8,9,10,11,12]
 assert task['loss_mask']==[0,0,0,0,0,1,1]
 assert all(r['metadata']['ignored_rejected_read_turns']==[1] for r in rows)
 assert all(r['metadata']['edge_tokens']==3 for r in rows)
 assert all(r['logprobs']==[] for r in rows)
