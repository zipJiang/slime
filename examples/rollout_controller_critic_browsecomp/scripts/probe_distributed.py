"""Exercise CUDA collectives on new training allocations before model loading."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=90))
    try:
        rank=dist.get_rank();world=dist.get_world_size()
        value=torch.tensor(float(rank+1),device='cuda')
        dist.all_reduce(value)
        assert value.item()==world*(world+1)/2
        for first in range(0,world,2):
            group=dist.new_group([first,first+1])
            if rank in (first,first+1):
                pair=torch.tensor(float(rank+1),device='cuda')
                dist.all_reduce(pair,group=group)
                assert pair.item()==2*first+3
        reports=[None]*world
        dist.all_gather_object(reports,dict(rank=rank,host=socket.gethostname(),
            local_rank=local_rank,gpu=torch.cuda.get_device_name(local_rank)))
        if rank==0:
            report=dict(passed=True,world_size=world,all_reduce=value.item(),
                tensor_parallel_pairs=world//2,ranks=reports)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
