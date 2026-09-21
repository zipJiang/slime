"""Model policies: preparation, generation, decoding and action parsing.

Token-native policies retain exact input and output IDs. Corporate API policies send
structured messages and report estimated IDs as inexact, suitable for evaluation.
"""

from step_controller.generation.openai_chat import OpenAIChatPolicy
from step_controller.generation.openai_responses import OpenAIResponsesPolicy
from step_controller.generation.policy import (
    Policy,
    PolicyFormat,
    PreparedPrompt,
    register_policy,
)
from step_controller.generation.sglang import SGLangPolicy
from step_controller.generation.slime import SlimePolicy
from step_controller.generation.tinker import TinkerPolicy
from step_controller.generation.types import (
    GenerateResult,
    NativeToolCall,
    SamplingParams,
    TokenId,
)
from step_controller.generation.vllm import VLLMPolicy

__all__ = [
    "OpenAIChatPolicy",
    "OpenAIResponsesPolicy",
    "SGLangPolicy",
    "SlimePolicy",
    "TinkerPolicy",
    "Policy",
    "PolicyFormat",
    "PreparedPrompt",
    "VLLMPolicy",
    "GenerateResult",
    "NativeToolCall",
    "SamplingParams",
    "TokenId",
    "register_policy",
]
