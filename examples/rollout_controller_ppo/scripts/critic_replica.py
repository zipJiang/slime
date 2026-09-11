"""Slime-owned frozen critic replica, with explicit publication barriers.

Only model tensors cross this boundary. Optimizers and checkpoint ownership stay
on the native Megatron ranks. A version cannot change during a scoring session.
"""
import hashlib
import json
import math
from pathlib import Path
import threading
import time


class FrozenCriticReplica:
    def __init__(self, base, max_length):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.max_length = max_length
        self.model = None
        self.version = None
        self.active = False

    def publish(self, directory, version):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel
        if self.active:
            raise RuntimeError('Cannot publish a critic during collection')
        start = time.monotonic()
        directory = Path(directory)
        report = json.loads((directory/'manifest.json').read_text())
        if report['version'] != version:
            raise ValueError('Snapshot version mismatch')
        for name, digest in report['sha256'].items():
            with (directory/name).open('rb') as stream:
                actual_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual_digest != digest:
                raise ValueError(f'Critic snapshot checksum mismatch: {name}')
        if self.model is None:
            self.model, info = AutoModel.from_pretrained(directory, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError(f'Critic backbone loading mismatch: {info}')
            self.model.eval().to('cuda')
        else:
            self.model.load_state_dict(load_file(str(directory/'model.safetensors')), strict=True)
        self.head = {k:v.to(device='cuda', dtype=torch.bfloat16) for k,v in
                     load_file(str(directory/'value_head.safetensors')).items()}
        if set(self.head) != {'weight', 'bias'} or self.head['weight'].shape[0] != 1:
            raise ValueError('Expected a scalar critic head with bias')
        torch.cuda.synchronize()
        self.version = version
        return dict(version=version, seconds=time.monotonic()-start,
                    device=torch.cuda.get_device_name(), manifest=report)

    def begin(self, version):
        if self.active or version != self.version or self.model is None:
            raise ValueError('Inactive, duplicate, or wrong critic snapshot')
        self.active = True
        return version

    def end(self, version):
        if not self.active or version != self.version:
            raise ValueError('Critic session mismatch')
        self.active = False
        return version

    def score(self, contexts, version):
        import torch
        if not self.active or version != self.version:
            raise ValueError('Stale or inactive critic version')
        if not contexts or any(not isinstance(c, str) or not c for c in contexts):
            raise ValueError('Expected nonempty contexts')
        scores = []
        with torch.inference_mode():
            for context in contexts:
                ids = self.tokenizer.encode(context, add_special_tokens=False)
                if not 0 < len(ids) < self.max_length:
                    raise ValueError('Critic context outside native sequence budget')
                tokens = torch.tensor([ids], device='cuda')
                hidden = self.model(input_ids=tokens, use_cache=False).last_hidden_state[0, -1]
                value = torch.nn.functional.linear(hidden, self.head['weight'], self.head['bias'])
                score = value.float().sigmoid().item()
                if not math.isfinite(score):
                    raise ValueError('Nonfinite critic prediction')
                scores.append(score)
        return dict(version=version, scores=scores)


class ReplicaScorer:
    """Same HTTP-facing interface as the synchronous native CriticScorer."""
    def __init__(self, replica):
        self.replica = replica
        self.version = None
        self.lock = threading.Lock()

    def begin(self, version):
        import ray
        with self.lock:
            if self.version is not None:
                raise RuntimeError('Critic session already active')
            if ray.get(self.replica.begin.remote(version)) != version:
                raise RuntimeError('Critic snapshot version mismatch')
            self.version = version

    def end(self):
        import ray
        with self.lock:
            if self.version is None:
                raise RuntimeError('No critic session')
            ray.get(self.replica.end.remote(self.version))
            self.version = None

    def score(self, contexts, version):
        import ray
        with self.lock:
            if self.version is None or self.version != version:
                raise ValueError('Stale or inactive critic version')
            return ray.get(self.replica.score.remote(contexts, version))
