"""Trainer-owned checkpoint value inference on Slime's critic ranks.

The driver brackets each collection with begin/end scoring. All ranks must be
called together and batches serialized: these methods execute TP collectives.
No optimizer update is allowed inside that bracket.
"""
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.data import get_data_iterator
from slime.backends.megatron_utils.model import forward_only, train
from targets import prepared_advantages
from value_loss import probability_values


class CheckpointCriticActor(MegatronTrainRayActor):
    _scoring_version = None

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        if args.use_wandb and not getattr(args, 'wandb_run_id', None):
            raise RuntimeError('Critic worker requires the initialized primary tracking run ID')
        # The base LM has no scalar value-head bias. Native HF loading already
        # skips the differently shaped head weight; explicitly initialize its
        # new bias instead of asking the Qwen LM mapper for a nonexistent tensor.
        import torch
        from slime.backends.megatron_utils import hf_to_megatron
        from slime.backends.megatron_utils.hf_to_megatron.common import strip_mcore_wrappers
        original = hf_to_megatron._LOADERS['qwen3_5']
        def critic_tensor(name, reader, config):
            if strip_mcore_wrappers(name).removeprefix('language_model.') == 'output_layer.bias':
                return torch.zeros(1)
            return original(name, reader, config)
        hf_to_megatron._LOADERS['qwen3_5'] = critic_tensor
        try:
            return super().init(args, role, with_ref, with_opd_teacher)
        finally:
            hf_to_megatron._LOADERS['qwen3_5'] = original

    def begin_scoring(self, version):
        if self.role != 'critic' or self._scoring_version is not None:
            raise RuntimeError('Invalid critic scoring transition')
        if not self.args.offload_train:
            raise ValueError('PPO critic sharing requires train offload')
        if self.args.context_parallel_size != 1:
            raise ValueError('Checkpoint inference currently requires CP=1')
        self.wake_up()
        self._scoring_version = version
        return version

    def score_checkpoints(self, data_ref, version):
        if self._scoring_version != version:
            raise RuntimeError('Critic inference version mismatch')
        data = self._get_rollout_data(data_ref)
        if any(n != 1 for n in data['response_lengths']):
            raise ValueError('Critic inference requires one checkpoint position')
        output = forward_only(probability_values, self.args, self.model,
                              get_data_iterator(data), data['num_microbatches'])
        values = output.get('values')
        if values is None:
            return dict(version=version, partition=[], values=[])
        return dict(version=version, partition=data['partition'],
                    values=[float(v.detach().float().cpu().item()) for v in values])

    def end_scoring(self, version):
        if self._scoring_version != version:
            raise RuntimeError('Critic inference version mismatch')
        self.sleep()
        self._scoring_version = None
        return version

    def train(self, rollout_id, rollout_data_ref, external_data=None):
        if self._scoring_version is not None:
            raise RuntimeError('Cannot train critic while a search uses its prior')
        return super().train(rollout_id, rollout_data_ref, external_data)

    def train_critic(self, rollout_id, data):
        if any(n != 1 for n in data['response_lengths']):
            raise ValueError('Actor sequences reached checkpoint critic training')
        iterator = get_data_iterator(data)
        data.update(forward_only(probability_values, self.args, self.model,
                                 iterator, data['num_microbatches']))
        prepared_advantages(self.args, data)
        self.args.ppo_critic_warmup = rollout_id < self.args.num_critic_only_steps
        self.args.loss_type = 'custom_loss'
        self.args.custom_loss_function_path = 'value_loss.checkpoint_loss'
        train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler,
              iterator, data['num_microbatches'], data['global_batch_sizes'])
        # These checkpoint values cannot be aligned with the actor-span batch.
        return {}
