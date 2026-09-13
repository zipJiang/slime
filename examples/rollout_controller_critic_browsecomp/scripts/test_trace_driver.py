"""Run the warmup driver lifecycle against deterministic native/replica stand-ins."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('improves', [True, False])
def test_checkpoint_selection_early_stop_and_model_only_reload(tmp_path, monkeypatch, improves):
    import trace_train_critic as m
    from trace_warmup import report

    out = tmp_path/'training'; collection=tmp_path/'collection'; collection.mkdir()
    experiment=tmp_path/'experiment'; (experiment/'data').mkdir(parents=True)
    (experiment/'data/split.json').write_text('{}')
    (collection/'manifest.json').write_text(json.dumps({'environment':m.environment_contract()}))
    (collection.parent/'collection-audit.json').write_text(json.dumps(dict(passed=True,full_collection=True,identities_exact=True)))
    records=lambda count:[dict(context=f'q{i}',target=float(i%2),group_index=str(i),turn=turn,folds=turn,
        metadata=dict(lane='critic',node_id=f'{i}/{turn}',target_source='monte_carlo_suffix',
                      diagnostics=dict(observations=1,mean_return=float(i%2)))) for i in range(count) for turn in (0,1)]
    data=dict(train=records(128),validation=records(8))
    source=dict(critic=dict(load='source',ckpt_step=15))
    args=SimpleNamespace(actor_num_nodes=3,actor_num_gpus_per_node=2,tensor_model_parallel_size=2,
        context_parallel_size=1,pipeline_model_parallel_size=1,global_batch_size=8,use_critic=True,
        offload_train=True,release_train=False,normalize_advantages=False,calculate_per_token_loss=False,
        critic_epochs=2,critic_eval_interval=4,seq_length=98304,num_critic_only_steps=32,
        trace_patience=2,lr=1e-6,trace_equivalence_tolerance=.01,trace_root_mass=.25,
        trace_source_candidate=tmp_path/'candidate.json',save=str(out/'native'),
        critic_collection=str(collection),hf_checkpoint='base')
    monkeypatch.setattr(m,'EXPERIMENT',experiment)
    monkeypatch.setattr(m,'configure_logger',lambda:None)
    monkeypatch.setattr(m,'verify_harness',lambda:None)
    monkeypatch.setattr(m,'require_pilot_candidate',lambda path:source)
    monkeypatch.setattr(m,'validate_collection_provenance',lambda *a:{})
    monkeypatch.setattr(m,'audit_sampling',lambda *a,**kw:{'passed':True})
    monkeypatch.setattr(m,'load_dataset',lambda *a:data)
    monkeypatch.setattr(m,'sha256',lambda *a:'a'*64)
    monkeypatch.setattr(m,'function_sha256',lambda *a:'b'*64)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=4*1024**4))
    monkeypatch.setattr(m,'placement',lambda n:('placement',[('train'+str(i//2),i%2) for i in range(6)]))
    monkeypatch.setattr(m,'partition_data',lambda args,parallel,data:data)
    monkeypatch.setattr(m,'put_packets',lambda data:data)
    monkeypatch.setattr(m,'probe_indices',lambda rows,lengths:list(range(len(rows))))
    tokenizer=SimpleNamespace(eos_token_id=0,encode=lambda *a,**kw:[1,2])
    monkeypatch.setattr(m.AutoTokenizer,'from_pretrained',lambda *a,**kw:tokenizer)
    monkeypatch.setattr(m,'NodeAffinitySchedulingStrategy',lambda *a,**kw:None)
    audited=[];groups=[];evaluated=[]
    monkeypatch.setattr(m,'audit_checkpoint',lambda path,steps,role:audited.append((path,steps,role)))
    monkeypatch.setattr(m,'build_candidate',lambda *a,**kw:dict(built=True,source=kw))
    def prediction(step):
        if not improves:return .5 if step==0 else .6
        return .8 if step==0 else (.5 if step==4 else .6 if step==8 else .7)
    class Remote:
        def __init__(self, fn):self.remote=fn
    class Group:
        def __init__(self,args):
            self.step=0 if args.load=='source' else args.ckpt_step+1
            self.released=False
            def export(directory,version):
                path=Path(directory)
                if not path.exists():
                    path.mkdir(parents=True);(path/'manifest.json').write_text(json.dumps(dict(version=version)))
                return {}
            self._actor_handlers=[SimpleNamespace(audit_optimizer_start=Remote(lambda:dict(fresh=True)),
                export_snapshot=Remote(export)) for _ in range(6)]
        def create(self):return [0]*6
        def async_train(self,step,packet):
            assert step==self.step
            assert len(packet['rollout_ids'])==16
            self.step+=1
            return None
        def save_model(self,iteration,force_sync):
            assert iteration+1==self.step and force_sync
            (Path(args.save)/f'iter_{iteration:07d}').mkdir(parents=True)
        def release(self):self.released=True
    def allocate(args,*a,**kw):
        group=Group(args);groups.append(group);return group
    monkeypatch.setattr(m,'allocate_train_group',allocate)
    class Scorer:
        def __init__(self,handlers,*a):
            self.group=next(g for g in groups if g._actor_handlers is handlers)
        def begin(self,version):self.version=version
        def end(self):self.version=None
        def score(self,contexts,version):
            assert version==self.version
            return dict(version=version,scores=[prediction(self.group.step)]*len(contexts))
    monkeypatch.setattr(m,'CriticScorer',Scorer)
    class Deferred:
        def __init__(self, fn):self.resolve=fn
    class Validator:
        def __init__(self):self.evaluate=Remote(lambda *a:Deferred(lambda:self.run(*a)))
        def run(self,directory,version,rows,indices,native,constant,mass,tolerance):
            step=int(version.rsplit('-',1)[1]);evaluated.append(step)
            assert native['scores']==[prediction(step)]*len(indices)
            scores=[prediction(step)]*len(rows)
            return dict(publication={},comparison=dict(passed=True,tolerance=tolerance),predictions=scores,
                        report=report(rows,scores,constant,mass))
    class Factory:
        def options(self,**kw):return self
        def remote(self,*a):return Validator()
    monkeypatch.setattr(m.ray,'get',lambda value:value.resolve() if isinstance(value,Deferred) else value)
    monkeypatch.setattr(m.ray,'put',lambda value:value)
    monkeypatch.setattr(m.ray,'kill',lambda *a:None)
    monkeypatch.setattr(m.ray,'nodes',lambda:[dict(Alive=True,Resources={'browsecomp_trace_validator':1},NodeManagerAddress='validator',NodeID='validator')])
    monkeypatch.setattr(m.ray,'remote',lambda **kw:lambda cls:Factory())
    m.run(args)
    complete=json.loads((out/'complete.json').read_text())
    assert evaluated==([0,4,8,12,16] if improves else [0,4,8,12])
    assert all(g.released for g in groups)
    if improves:
        assert complete['native_iteration']==3
        assert complete['updates']==4 and complete['optimizer_updates_executed']==16
        assert len(groups)==2 and groups[-1].step==4
        assert audited[0][1:]==(4,'critic')
        assert (out/'warmstart-candidate.json').exists()
        assert json.loads((out/'reload-audit.json').read_text())['cursors']==[0]*6
    else:
        assert complete['selected_update']==0 and not complete['ready_for_joint_training']
        assert len(groups)==1 and not audited
        assert not (out/'warmstart-candidate.json').exists()
