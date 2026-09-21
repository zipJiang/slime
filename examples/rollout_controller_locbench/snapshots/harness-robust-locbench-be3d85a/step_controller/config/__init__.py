"""Optional config helpers for experiment and CLI entry points.

:class:`~step_controller.config.rollout.RolloutConfig` is the authorable surface of a
rollout -- every knob a config file may say -- and :class:`RolloutSpec` is that record
plus the two inputs an actor's factory needs (``env``, ``system_prompt``). Both build
themselves from a mapping through the registry.

Implementations register themselves through the registry ``register`` free
function (e.g. ``register(Policy, "name", ...)``); there are no per-type
registration wrappers here. ``load_yaml`` is re-exported here for convenience; it
imports OmegaConf inside the call, so this package still costs nothing to import
without the ``config`` extra.
"""

from step_controller.config.omegaconf import load_yaml
from step_controller.config.rollout import (
    RolloutConfig,
    RolloutSpec,
    SearchPass,
    VinePpoConfig,
)

__all__ = [
    "SearchPass",
    "VinePpoConfig",
    "RolloutConfig",
    "RolloutSpec",
    "load_yaml",
]
