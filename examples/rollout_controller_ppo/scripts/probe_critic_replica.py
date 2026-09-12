"""Read-only, one-GPU ablations against an archived native critic comparison.

This never updates weights or publishes a critic to training. It checks tensor
identity and exact context hashes before comparing alternative inference paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--contexts', type=Path, required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from critic_replica import FrozenCriticReplica

    audit = json.loads(args.audit.read_text())
    version = audit['version']
    replica = FrozenCriticReplica(args.base, 32768)
    publication = replica.publish(args.snapshot, version)
    assert publication['manifest']['sha256'] == audit['publication']['manifest']['sha256']
    tokenizer = replica.tokenizer
    contexts = [tokenizer.apply_chat_template([dict(role='user', content=s)],
        tokenize=False, add_generation_prompt=True) for s in
        ('Checkpoint scoring readiness.', 'A different checkpoint.')]
    contexts += [r['context'] for r in json.loads(args.contexts.read_text())['contexts']]
    contexts += [tokenizer.apply_chat_template([dict(role='user',
        content='A passenger has requested a change. '*n)], tokenize=False,
        add_generation_prompt=True) for n in (1000, 3000)]
    ids = [tokenizer.encode(c, add_special_tokens=False) for c in contexts]
    assert [hashlib.sha256(c.encode()).hexdigest() for c in contexts] == audit['context_sha256']
    assert [len(t) for t in ids] == audit['context_tokens']
    result = dict(version=version, audit=str(args.audit),
        audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        publication=publication, torch=torch.__version__, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)

    def run(mode, *, sentinel=False, pad=False, packed=False):
        start = time.monotonic()
        with torch.inference_mode():
            for i, original in enumerate(ids):
                tokens = original + ([tokenizer.eos_token_id] if sentinel else [])
                if pad:
                    tokens += [tokenizer.eos_token_id] * (-len(tokens) % 128)
                inputs = torch.tensor([tokens], device='cuda')
                kwargs = {}
                if packed:
                    kwargs = dict(cu_seq_lens_q=torch.tensor([0, len(tokens)],
                                  dtype=torch.int32, device='cuda'))
                hidden = replica.model(input_ids=inputs, use_cache=False, **kwargs).last_hidden_state
                position = len(original)-1
                linear = torch.nn.functional.linear
                weight, bias = replica.head['weight'], replica.head['bias']
                values = dict(
                    final_head=linear(hidden[0, position], weight, bias),
                    full_head=linear(hidden, weight, bias)[0, position],
                    fp32_head=linear(hidden[0, position].float(), weight.float(), bias.float()))
                scores = {k: v.float().sigmoid().item() for k, v in values.items()}
                native = audit['native']['scores'][i]
                result['rows'].append(dict(mode=mode, index=i, tokens=len(original),
                    input_tokens=len(tokens), native=native, scores=scores,
                    errors={k: abs(v-native) for k, v in scores.items()}))
                save()
                print(json.dumps(result['rows'][-1]), flush=True)
        result.setdefault('seconds', {})[mode] = time.monotonic()-start
        save()

    # Isolate input extent, head GEMM shape, and packed GDN execution first.
    run('sdpa_context')
    run('sdpa_sentinel', sentinel=True)
    run('sdpa_padded', sentinel=True, pad=True)
    run('sdpa_packed_gdn', sentinel=True, packed=True)

    # Native Slime uses FLA's ShortConvolution and gated norm. Substitute only
    # those linear-attention blocks, retaining the exact exported tensors.
    from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

    class NativeBlock(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            if cache_params is not None or attention_mask is not None:
                raise ValueError('Probe supports uncached, unmasked singleton sequences only')
            cu = torch.tensor([0, hidden_states.shape[1]], device=hidden_states.device,
                              dtype=torch.int32)
            return self.block(hidden_states, cu_seqlens=cu)

    for i, layer in enumerate(replica.model.layers):
        if not hasattr(layer, 'linear_attn'):
            continue
        old = layer.linear_attn
        state = old.state_dict()
        native = Qwen3_5GatedDeltaNet(replica.model.config, i).to(device='cuda', dtype=torch.bfloat16)
        state['conv1d.weight'] = state['conv1d.weight'].squeeze(1)
        native.load_state_dict(state, strict=True)
        for name, tensor in native.state_dict().items():
            assert torch.equal(tensor, state[name]), name
        layer.linear_attn = NativeBlock(native).eval()
        del old, state
    torch.cuda.empty_cache()
    run('native_gdn_sdpa_sentinel', sentinel=True)
    result['complete'] = True
    save()


if __name__ == '__main__':
    main()
