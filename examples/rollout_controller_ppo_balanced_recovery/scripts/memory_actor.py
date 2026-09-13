"""Native PPO actor with opt-in host storage of Adam moments between updates."""
import json

from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from optimizer_memory import offload_actor_moments


class MemoryBoundActor(MegatronTrainRayActor):
    def train_actor(self, rollout_id, rollout_data, external_data=None):
        with offload_actor_moments(self.optimizer) as report:
            print('actor-moment-offload '+json.dumps(dict(rollout_id=rollout_id, **report)), flush=True)
            return super().train_actor(rollout_id, rollout_data, external_data)
