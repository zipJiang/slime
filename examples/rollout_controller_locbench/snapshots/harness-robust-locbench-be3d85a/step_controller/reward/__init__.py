"""Reward: sequence-level reward for a serialized state context.

Mirrors :mod:`step_controller.generation`: a registry-pluggable :class:`RewardModel`
interface (sync + async) with backends registered via ``register_reward``. The first
backend, :class:`VLLMRewardModel`, scores against a vLLM pooling/reward model over
``POST /pooling``.
"""

from step_controller.reward.config import RewardConfig
from step_controller.reward.interfaces import (
    AsyncRewardModel,
    RewardModel,
    register_reward,
    run_async,
)
from step_controller.reward.types import RewardResult
from step_controller.reward.vllm import VLLMRewardModel

__all__ = [
    "AsyncRewardModel",
    "RewardConfig",
    "RewardModel",
    "RewardResult",
    "VLLMRewardModel",
    "register_reward",
    "run_async",
]
