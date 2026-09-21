"""Native actor SFT with held-out masked token likelihood evaluation."""
from fresh_actor import FreshStartActor
from slime.backends.megatron_utils.data import get_data_iterator


class ImitationActor(FreshStartActor):
    def evaluate_targets(self,data_ref):
        if self.args.offload_train:self.wake_up()
        try:
            data=self._get_rollout_data(data_ref)
            result=self.compute_log_prob(get_data_iterator(data),data['num_microbatches'])
            if 'log_probs' not in result:return []
            import torch
            rows=[]
            for index,logps,mask in zip(data['partition'],result['log_probs'],data['loss_masks'],strict=True):
                logps=logps.float()
                mask=torch.as_tensor(mask,device=logps.device,dtype=logps.dtype)
                if logps.numel()!=mask.numel():raise ValueError('Evaluation target alignment mismatch')
                rows.append(dict(index=index,nll=float(-(logps*mask).sum().float().cpu()),tokens=int(mask.sum())))
            return rows
        finally:
            if self.args.offload_train:self.sleep()
