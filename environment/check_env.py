"""Verify the overlay and core Qwen3.5 training imports, optionally on GPUs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from packaging.requirements import Requirement

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve() == (root/'.venv').resolve(), sys.prefix
import torch
import ray
import slime
import megatron.core
import transformer_engine.pytorch
from slime_plugins.models.qwen3_5 import get_qwen3_5_spec
from slime.backends.megatron_utils.loss import compute_advantages_and_returns
assert Path(slime.__file__).resolve().is_relative_to(root), slime.__file__
# uv pip check inspects only the overlay, not inherited image distributions.
# Check Slime's declared runtime requirements using Python's actual import path.
for raw in importlib.metadata.requires('slime') or []:
    requirement = Requirement(raw)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    version = importlib.metadata.version(requirement.name)
    assert version in requirement.specifier, (requirement.name, version, raw)
packages = ('torch','transformers','ray','slime','megatron-core',
            'transformer-engine','flash-linear-attention','sglang','torch-memory-saver')
report = dict(python=sys.version, executable=sys.executable,
              packages={name: importlib.metadata.version(name) for name in packages},
              slime_source=slime.__file__, megatron_source=megatron.core.__file__,
              cuda_runtime=torch.version.cuda, gpu_test=False)
if args.gpu:
    assert torch.cuda.is_available(), 'CUDA unavailable in SIF'
    report['gpus'] = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            x = torch.randn(32,32,device='cuda',requires_grad=True)
            (x@x).square().mean().backward()
            torch.cuda.synchronize()
            assert torch.isfinite(x.grad).all()
            report['gpus'].append(torch.cuda.get_device_name(index))
    report['gpu_test'] = True
text = json.dumps(report,indent=2)+'\n'
if args.output:
    args.output.write_text(text)
print(text)
