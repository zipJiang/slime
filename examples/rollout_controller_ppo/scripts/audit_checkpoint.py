"""Read every tensor in a Torch distributed checkpoint on one CPU process.

This validates storage and tensor contents. Native optimizer step records and
HF model loading are checked separately. Run inside the pinned Slime SIF.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


class ExactKeyPlanner(_EmptyStateDictLoadPlanner):
    def _should_include_key(self, key, metadata):
        # Megatron saves flat keys without Torch's optional planner_data.
        return key in self.keys


def walk(value, prefix=''):
    if isinstance(value,dict):
        for key,item in value.items():
            yield from walk(item,f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value,(list,tuple)):
        for key,item in enumerate(value):
            yield from walk(item,f'{prefix}.{key}')
    elif isinstance(value,torch.Tensor):
        yield prefix,value


def audit(directory, expected_steps, role):
    reader = FileSystemReader(directory)
    metadata = reader.read_metadata()
    keys = sorted(metadata.state_dict_metadata)
    heads = [value for key, value in metadata.state_dict_metadata.items()
             if key.endswith('output_layer.weight')]
    if len(heads) != 1 or (heads[0].size[0] == 1) != (role == 'critic'):
        raise ValueError('Checkpoint head does not match expected actor/critic role')
    # Megatron's DCP keys are module-relative (decoder/embedding/output_layer),
    # while optimizer buffers retain their optimizer namespace.
    if not any(k.startswith(('model.', 'decoder.', 'embedding.', 'output_layer.')) for k in keys) or not any(k.startswith('optimizer.') for k in keys):
        raise ValueError('Expected both full-model and optimizer checkpoint entries')
    common = torch.load(directory/'common.pt', map_location='cpu', weights_only=False)
    common_evidence = {k:common[k] for k in ('iteration', 'checkpoint_version',
        'opt_param_scheduler', 'optimizer')}
    for storage in metadata.storage_data.values():
        file = directory/storage.relative_path
        if file.stat().st_size < storage.offset+storage.length:
            raise ValueError(f'Truncated checkpoint storage: {file}')
    counts = Counter()
    entries = []
    optimizer_states = {}
    moment_keys = []
    moments_nonzero = False
    for key in keys:
        state = {}
        _load_state_dict(state, storage_reader=reader,
            planner=ExactKeyPlanner(keys={key}),no_dist=True)
        if '.optimizer/shard' in key:
            optimizer_states.update(state)
        tensors = list(walk(state))
        for name,tensor in tensors:
            if key.endswith(('.exp_avg', '.exp_avg_sq')):
                moment_keys.append(key)
                if expected_steps == 0 and torch.count_nonzero(tensor).item():
                    moments_nonzero = True
            # Bound the temporary boolean allocation for large optimizer buffers.
            flat = tensor.reshape(-1)
            for chunk in flat.split(4_000_000):
                if not torch.isfinite(chunk).all().item():
                    raise ValueError(f'Nonfinite checkpoint tensor: {name}')
            counts['tensors'] += 1
            counts['tensor_bytes'] += tensor.numel()*tensor.element_size()
        entries.append(dict(key=key,tensors=len(tensors),
                            bytes=sum(t.numel()*t.element_size() for _,t in tensors)))
        del state,tensors
        tensor = flat = chunk = None
    optimizer_steps = sorted({group.get('step', 0) or 0 for shards in optimizer_states.values()
                              for shard in shards for group in shard['param_groups']})
    if optimizer_steps != [expected_steps]:
        raise ValueError(f'Optimizer steps {optimizer_steps} != expected role updates {expected_steps}')
    if expected_steps and not any(k.endswith('.exp_avg') for k in moment_keys):
        raise ValueError('Trained optimizer checkpoint is missing first moments')
    if expected_steps and not any(k.endswith('.exp_avg_sq') for k in moment_keys):
        raise ValueError('Trained optimizer checkpoint is missing second moments')
    if expected_steps == 0 and moments_nonzero:
        raise ValueError('An untrained actor has nonzero Adam moments')
    result = dict(checkpoint=str(directory.resolve()),role=role,entries=entries,counts=dict(counts),
                  full_storage_read=True,finite_tensors=True,common_state=common_evidence,
                  expected_optimizer_steps=expected_steps, moment_keys=sorted(set(moment_keys)),
                  optimizer_steps=optimizer_steps,optimizer_metadata=optimizer_states)
    (directory.parent/f'{directory.name}-readback.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checkpoint=str(directory),entries=len(entries),counts=dict(counts))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--expected-steps',type=int,required=True)
    parser.add_argument('--role',choices=['actor','critic'],required=True)
    torch.set_num_threads(4)
    args=parser.parse_args()
    audit(args.checkpoint, args.expected_steps, args.role)
