"""Publish actor weights after reference loading on non-offloaded Vine trainers."""
import json
from pathlib import Path
from vine_actor import VineActor as OriginalVineActor

class VineActor(OriginalVineActor):
    def update_weights(self):
        # Slime init loads the reference last. Without offload, it leaves the
        # reference active. The distributed updater reads live model parameters,
        # not the actor CPU backup, so explicitly restore actor before publishing.
        before = self._active_model_tag
        if not self.args.offload_train and before != 'actor':
            self._switch_model('actor')
        if not self.args.offload_train and self._active_model_tag != 'actor':
            raise RuntimeError('Cannot publish a non-actor model to rollout engines')
        result = super().update_weights()
        import torch.distributed as dist
        path = Path(self.args.save).parent/'actor-publication-audit'
        path.mkdir(exist_ok=True)
        version = self.weight_updater.weight_version
        (path/f'version-{version}-rank-{dist.get_rank()}.json').write_text(json.dumps(
            dict(previous_active_model=before, published_model='actor',
                 active_model=self._active_model_tag, weight_version=version), indent=2))
        return result
