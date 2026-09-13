"""Compare saved critic predictions across packing and train/eval execution.

Runs only on a separate, model-only checkpoint load. The training path receives
zero loss and both optimizer and scheduler steps are replaced with no-ops.
No checkpoint or model export is written.
"""
import hashlib
import json
import os
from pathlib import Path

import ray
import torch
from transformers import AutoTokenizer

from batches import partition_data, put_packets, training_data
from critic_actor import CheckpointCriticActor
from dataset import constant_baseline, load_dataset, metrics
from targets import checkpoint_fields, prepared_advantages
from train_critic import custom_args, placement, write
from value_loss import probability_values
from value_service import CriticScorer
from slime.backends.megatron_utils.data import get_data_iterator
from slime.backends.megatron_utils.model import forward_only, train
from slime.ray.placement_group import allocate_train_group
from slime.utils.arguments import parse_args

_measurements = []


def diagnostic_loss(args, batch, logits, reducer):
    _, output = probability_values(logits, args=args,
        unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
        response_lengths=batch['response_lengths'])
    values = torch.cat(output['values']).flatten()
    targets = torch.cat(batch['returns']).flatten()
    old = torch.cat(batch['values']).flatten()
    squared = reducer((values-targets).square())
    _measurements.append(dict(count=values.numel(),
        weighted_squared=float(squared.detach()),
        weighted_prediction=float(reducer(values).detach()),
        weighted_target=float(reducer(targets).detach()),
        max_train_eval_error=float((values-old).abs().max().detach())))
    return squared * 0, dict(diagnostic_mse=squared.detach())


class DiagnosticActor(CheckpointCriticActor):
    def compare_training(self, refs):
        from megatron.core import mpu
        if not (self.args.finetune and self.args.no_load_optim and self.args.no_load_rng):
            raise ValueError('Diagnostic requires an isolated model-only load')
        self.wake_up()
        data = self._get_rollout_data(refs)
        iterator = get_data_iterator(data)
        data.update(forward_only(probability_values, self.args, self.model,
                                iterator, data['num_microbatches']))
        prepared_advantages(self.args, data)
        self.args.ppo_critic_warmup = True
        self.args.loss_type = 'custom_loss'
        self.args.custom_loss_function_path = 'diagnose_critic.diagnostic_loss'
        import diagnose_critic
        diagnose_critic._measurements.clear()
        original_step = self.optimizer.step
        original_schedule = self.opt_param_scheduler.step
        skipped = []
        before = self.audit_optimizer_start()
        def no_update(*args, **kwargs):
            skipped.append(True)
            return True, 0., 0
        self.optimizer.step = no_update
        self.opt_param_scheduler.step = lambda *args, **kwargs: None
        try:
            train(0, self.model, self.optimizer, self.opt_param_scheduler,
                  iterator, data['num_microbatches'], data['global_batch_sizes'])
            after = self.audit_optimizer_start()
            if before != after or not skipped or not after['fresh']:
                raise ValueError('Diagnostic changed optimizer history')
            return dict(tp_rank=mpu.get_tensor_model_parallel_rank(),
                dp_rank=mpu.get_data_parallel_rank(), optimizer_before=before,
                optimizer_after=after, blocked_updates=len(skipped),
                microbatches=list(diagnose_critic._measurements))
        finally:
            self.optimizer.step = original_step
            self.opt_param_scheduler.step = original_schedule
            self.sleep()


def run(args):
    experiment = Path(__file__).resolve().parents[1]
    out = Path(args.save).parent
    out.mkdir(parents=True, exist_ok=True)
    args.save = None
    args.save_hf = None
    args.use_kl_loss = False
    args.kl_coef = 0
    args.use_opd = False
    args.disable_param_buffers_cpu_backup = False
    args.custom_advantage_function_path = None
    args.rollout_data_postprocess_path = None
    args.custom_tis_function_path = None
    args.init_method_std = .001
    if not (args.finetune and args.no_load_optim and args.no_load_rng):
        raise ValueError('Explicit model-only load flags required')
    data = load_dataset(args.critic_collection, json.loads((experiment/'data/split.json').read_text()))
    baseline = constant_baseline(data['train'])
    questions = sorted({r['group_index'] for r in data['train']},
        key=lambda q: hashlib.sha256(('critic-epoch0/'+q).encode()).hexdigest())[-8:]
    selected = [r for q in questions for r in data['train'] if r['group_index'] == q]
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, local_files_only=True)
    parallel = dict(dp_size=args.actor_num_nodes, cp_size=1, vpp_size=1,
                    microbatch_group_size_per_vp_stage=1)
    pg, physical = placement(args.actor_num_nodes)
    critic = allocate_train_group(args, args.actor_num_nodes, 2, pg,
                                 role='critic', actor_cls=DiagnosticActor)
    cursors = critic.create()
    if cursors != [0]*(2*args.actor_num_nodes):
        raise ValueError('Model-only load retained training cursor')
    write(out/'initialized.json', dict(physical=physical, cursors=cursors,
        contexts=len(selected), questions=questions, load=args.load, iteration=args.ckpt_step))
    scorer = CriticScorer(critic._actor_handlers, args, parallel, tokenizer, tokenizer.eos_token_id)
    version = 'saved-critic-diagnostic'
    scorer.begin(version)
    try:
        small = []
        for start in range(0, len(selected), 8):
            small.extend(scorer.score([r['context'] for r in selected[start:start+8]], version)['scores'])
        rows = [checkpoint_fields(dict(r, group_index=questions.index(r['group_index'])), tokenizer,
            sentinel_token_id=tokenizer.eos_token_id, max_sequence_length=args.seq_length,
            warmup=True) for r in selected]
        packets = put_packets(partition_data(args, parallel,
            training_data(rows, lane='critic', expected_groups=range(8))))
        reports = ray.get([a.score_checkpoints.remote(packets, version) for a in critic._actor_handlers])
        combined = {}
        for report in reports:
            for index, value in zip(report['partition'], report['values'], strict=True):
                if index in combined and abs(combined[index]-value)>1e-5:
                    raise ValueError('TP prediction mismatch')
                combined[index] = value
        packed = [combined[i] for i in range(len(selected))]
    finally:
        scorer.end()
    result = dict(small_pack=metrics(selected, small, baseline),
        training_pack=metrics(selected, packed, baseline),
        max_packing_error=max(abs(a-b) for a,b in zip(small, packed, strict=True)),
        rows=[dict(question=r['group_index'], target=r['target'],
            context_sha256=hashlib.sha256(r['context'].encode()).hexdigest(), small=a, packed=b)
            for r,a,b in zip(selected,small,packed,strict=True)])
    write(out/'packing.json', result)
    reports = ray.get([a.compare_training.remote(packets) for a in critic._actor_handlers])
    unique = [r for r in reports if r['tp_rank']==0]
    if len(unique)!=args.actor_num_nodes:
        raise ValueError('Missing DP diagnostic reports')
    microbatches = [m for r in unique for m in r['microbatches']]
    result['training_execution'] = dict(
        mse=sum(m['weighted_squared'] for m in microbatches)/8,
        predicted_mean=sum(m['weighted_prediction'] for m in microbatches)/8,
        target_mean=sum(m['weighted_target'] for m in microbatches)/8,
        max_train_eval_error=max(m['max_train_eval_error'] for m in microbatches), reports=reports)
    write(out/'complete.json', result)
    critic.release()
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    args = parse_args(custom_args)
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars':{
        'GLOO_SOCKET_IFNAME':'ens0','NCCL_SOCKET_IFNAME':'ens0','NCCL_IB_DISABLE':'1',
        'PYTHONPATH':os.environ['PYTHONPATH']}})
    run(args)
