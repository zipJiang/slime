"""Preserve untrained vision/MTP tensors and audit Slime's text-model HF export.

Run after Slime finishes saving. Missing trained text tensors are fatal. This
never substitutes source weights for a missing trained parameter.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def finalize(source, target, expect_unchanged=False):
    src = json.loads((source/'model.safetensors.index.json').read_text())
    index_path = target/'model.safetensors.index.json'
    out = json.loads(index_path.read_text())
    expected, actual = set(src['weight_map']), set(out['weight_map'])
    missing = expected-actual
    if actual-expected:
        raise ValueError(f'Unexpected exported keys: {sorted(actual-expected)}')
    trained_missing = [k for k in missing if not k.startswith(('model.visual.', 'mtp.'))]
    if trained_missing:
        raise ValueError(f'Missing trained text weights: {trained_missing}')
    preserved = {}
    for filename in sorted({src['weight_map'][k] for k in missing}):
        with safe_open(source/filename, framework='pt', device='cpu') as f:
            for key in sorted(missing):
                if src['weight_map'][key] == filename:
                    preserved[key] = f.get_tensor(key).clone()
    if preserved:
        filename = 'preserved-vision-mtp.safetensors'
        save_file(preserved, target/filename, metadata={'format':'pt'})
        out['weight_map'].update({k:filename for k in preserved})
    del preserved
    # Read back every tensor, check shape/dtype/finiteness, and identify updates.
    counts = defaultdict(int)
    changed = []
    dtype_changes = []
    total_size = 0
    sources = {name:safe_open(source/name,framework='pt',device='cpu')
               for name in set(src['weight_map'].values())}
    try:
        for filename in sorted(set(out['weight_map'].values())):
            with safe_open(target/filename,framework='pt',device='cpu') as f:
                indexed = {k for k,v in out['weight_map'].items() if v == filename}
                if set(f.keys()) != indexed:
                    raise ValueError(f'Shard/index disagreement in {filename}')
                for key in sorted(indexed):
                    tensor = f.get_tensor(key)
                    original = sources[src['weight_map'][key]].get_tensor(key)
                    if tensor.shape != original.shape:
                        raise ValueError(f'Incompatible shape: {key}')
                    if tensor.dtype != original.dtype:
                        # Slime trains these text parameters in BF16 alongside
                        # the other text weights; the initial HF files use FP32.
                        allowed = (key.startswith('model.language_model.layers.')
                            and key.endswith(('.linear_attn.A_log', '.linear_attn.norm.weight'))
                            and original.dtype == torch.float32 and tensor.dtype == torch.bfloat16)
                        if not allowed:
                            raise ValueError(f'Unexpected dtype change: {key}')
                        dtype_changes.append(dict(key=key,source=str(original.dtype),
                                                  exported=str(tensor.dtype)))
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError(f'Nonfinite tensor: {key}')
                    # Do not count initial dtype conversion as an optimizer update.
                    equal = torch.equal(tensor, original.to(tensor.dtype))
                    untrained = key.startswith(('model.visual.','mtp.'))
                    if untrained and not equal:
                        raise ValueError(f'Untrained tensor changed: {key}')
                    counts['untrained_preserved' if untrained else 'text_tensors'] += 1
                    if not equal:
                        changed.append(key)
                    total_size += tensor.numel()*tensor.element_size()
    finally:
        sources.clear()
    if expect_unchanged and changed:
        raise ValueError('Actor weights changed during critic-only warm-up')
    if not expect_unchanged and not changed:
        raise ValueError('No full-model text weight changed after actor training')
    out['metadata']['total_size'] = total_size
    temporary = index_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(out,indent=2)+'\n')
    temporary.replace(index_path)
    hashes={name:hashlib.file_digest((target/name).open('rb'),'sha256').hexdigest()
            for name in sorted(set(out['weight_map'].values()))}
    report=dict(source=str(source.resolve()), target=str(target.resolve()),
                counts=dict(counts), restored_keys=sorted(missing), changed_keys=changed,
                expect_unchanged=expect_unchanged,
                total_size=total_size, shard_sha256=hashes,dtype_changes=dtype_changes)
    (target/'export-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(counts=dict(counts),restored=len(missing),changed=len(changed),bytes=total_size)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--expect-unchanged',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    finalize(args.source,args.target,args.expect_unchanged)
