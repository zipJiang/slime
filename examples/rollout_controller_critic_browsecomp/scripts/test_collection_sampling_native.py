from types import SimpleNamespace
from pathlib import Path
import sys

EXPERIMENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXPERIMENT/'snapshots/harness'))

from audit_sampling import seed
from examples.deontic.synthetic_compaction import FOLD_DRAW, FoldGenerator
from step_controller.generation.vllm import VLLMPolicy


def test_audited_seed_matches_live_generator_and_advances(monkeypatch):
    params=SimpleNamespace(temperature=1.0,top_p=1.0)
    monkeypatch.setattr(VLLMPolicy,'_prepare',
        lambda self,prefix,sampling:(tuple(prefix),params,dict(max_tokens=1)))
    generator=object.__new__(FoldGenerator)
    key='browsecomp-critic-v1/train/question/2'
    token=FOLD_DRAW.set((key,0))
    try:
        first=generator._prepare([1],params)[2]
        second=generator._prepare([1],params)[2]
    finally:
        FOLD_DRAW.reset(token)
    assert first['seed']==seed('train','question',2,0)
    assert second['seed']==seed('train','question',2,1)
    assert first['seed']!=second['seed']
