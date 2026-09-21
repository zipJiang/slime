"""Exercise native packing and loss autograd; no model or optimizer is allocated."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ppo_batches import actor_packets
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean


def main():
    args=SimpleNamespace(global_batch_size=4,calculate_per_token_loss=False,
        use_dynamic_batch_size=True,max_tokens_per_gpu=9,balance_data=True,balance_by_flops=False,micro_batch_size=1,
        hidden_size=4096,num_attention_heads=16,num_query_groups=4,kv_channels=256,
        vocab_size=248320,ffn_hidden_size=12288,num_experts=None,num_layers=32)
    cases=[]
    for dp,count in [(1,1),(2,1),(2,3),(4,3)]:
        rows=[dict(tokens=list(range(9)),response_length=8,reward=.2,loss_mask=[1]*8,
            rollout_log_probs=[-.1]*8,group_index=1,
            metadata=dict(lane='actor',tag='task' if i%2==0 else 'fold',node_id=i,edge_tokens=8,group_edge_count=7)) for i in range(count)]
        parallel=dict(dp_size=dp,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
        packets,report=actor_packets(args,parallel,rows,expected_groups=range(4))
        parameter=torch.tensor(.7,requires_grad=True);loss=0
        for packet in packets:
            assert packet['global_batch_sizes']==[4]
            assert len(packet['sampling_temperatures'])==len(packet['tokens'])
            assert set(packet['sampling_temperatures'])<={.6,.7}
            with patch('slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_world_size',return_value=1):
                reducer=get_sum_of_sample_mean([len(t) for t in packet['tokens']],packet['response_lengths'],
                    packet['loss_masks'],packet['rollout_mask_sums'],False)
                # Summed DP/microbatch contributions after Megatron's accumulation
                # and DP averaging; the original four-question normalizer remains.
                loss=loss+reducer(parameter.expand(sum(packet['response_lengths'])))/4
        loss.backward()
        expected=count/7/4
        assert torch.allclose(parameter.grad,torch.tensor(expected)),(dp,count,parameter.grad,expected)
        cases.append(dict(dp=dp,source_records=count,gradient=parameter.grad.item(),**report))
    print(json.dumps(dict(passed=True,cases=cases,scope='Real native packer/tensorizer and CP1 scalar loss autograd; no distributed optimizer or model benchmark')),flush=True)


if __name__=='__main__':main()
