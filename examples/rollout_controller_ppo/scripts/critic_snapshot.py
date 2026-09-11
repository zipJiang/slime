"""Model-only native critic export, called collectively at a drained boundary."""
import hashlib
import json
from pathlib import Path
import time


def export_snapshot(worker, directory, version):
    import torch
    import torch.distributed as dist
    from megatron.core import mpu
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
    from slime.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers
    if worker.role != 'critic' or worker._scoring_version is not None:
        raise RuntimeError('Critic export requires an inactive scoring session')
    if worker.args.pipeline_model_parallel_size != 1 or worker.args.context_parallel_size != 1:
        raise ValueError('Critic snapshot currently requires PP=CP=1')
    start = time.monotonic()
    worker.wake_up()
    rank = dist.get_rank()
    directory = Path(directory)
    backbone, head = {}, {}
    try:
        # Only the first DP replica gathers tensors; all of its TP ranks join.
        if mpu.get_data_parallel_rank() == 0:
            for name, param in named_params_and_buffers(worker.args, worker.model):
                tensor = all_gather_param(name, param)
                if dist.get_rank() != 0:
                    continue
                if name.endswith(('output_layer.weight', 'output_layer.bias')):
                    head[name.rsplit('.', 1)[1]] = tensor.detach().cpu().contiguous()
                    continue
                for key, value in convert_to_hf(worker.args, 'qwen3_5', name, tensor):
                    if not key.startswith('model.language_model.'):
                        raise ValueError(f'Unexpected critic tensor: {key}')
                    key = key.removeprefix('model.language_model.')
                    if key in backbone:
                        raise ValueError(f'Duplicate critic tensor: {key}')
                    backbone[key] = value.detach().cpu().contiguous()
        if dist.get_rank() == 0:
            directory.mkdir(parents=True, exist_ok=False)
            if set(head) != {'weight', 'bias'} or head['weight'].shape[0] != 1:
                raise ValueError('Expected a scalar critic head with bias')
            if any(not torch.isfinite(t).all() for t in [*backbone.values(), *head.values()]):
                raise ValueError('Nonfinite critic snapshot')
            save_file(backbone, str(directory/'model.safetensors'))
            save_file(head, str(directory/'value_head.safetensors'))
            AutoConfig.from_pretrained(worker.args.hf_checkpoint, local_files_only=True).text_config.save_pretrained(directory)
            hashes = {}
            for name in ('model.safetensors', 'value_head.safetensors', 'config.json'):
                with (directory/name).open('rb') as stream:
                    hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest = dict(version=version, sha256=hashes, backbone_tensors=len(backbone),
                            model_only=True, seconds=time.monotonic()-start)
            (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        dist.barrier()
    finally:
        worker.sleep()
    return dict(rank=rank, version=version, seconds=time.monotonic()-start)
