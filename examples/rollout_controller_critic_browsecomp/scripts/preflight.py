"""Exercise the actual BrowseComp state/serializer without requesting a GPU."""
import asyncio
import json
from pathlib import Path
import sys

EXPERIMENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXPERIMENT/'snapshots/harness'))
from collect import context
from examples.browsercomp_plus.env import Case, RetrievalArchive, make_runner
from step_controller.codec import ChatCodec
from step_controller.generation import Policy, PolicyFormat, SamplingParams, GenerateResult
from step_controller.generation.parsing import QwenXMLToolCallParser
from transformers import AutoTokenizer


class SubmitPolicy(Policy):
    async def agenerate_tokens(self,prefix_tokens,sampling_params=None):
        text='done</think>\n<tool_call>\n<function=submit>\n<parameter=answer>Paris</parameter>\n</function>\n</tool_call>'
        return GenerateResult(tokens=tuple(self.format.codec.encode(text)),
                              prefix_tokens=tuple(prefix_tokens),text=text,exact_generation=False)


async def main():
    base='/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    codec=ChatCodec(tokenizer)
    policy=SubmitPolicy(format=PolicyFormat('qwen_xml',codec,QwenXMLToolCallParser()),
                        default_params=SamplingParams(max_tokens=6144,temperature=1,top_p=1))
    case=Case('synthetic','Which city?',('Paris',))
    runner,replay,workspace,prompt,compactor,tools=make_runner(case,codec,policy,archive=RetrievalArchive())
    replay.allow_live=True
    state=await runner.start(prompt,workspace=workspace)
    serialized=context(state.snapshot(),tools,tokenizer)
    assert 'Paris' not in serialized
    assert 'remaining_task_steps' in serialized and 'calls_remaining' in serialized
    state=await runner.advance(state)
    assert state.done and state.state.answer=='Paris'
    forced=await runner.finish(await runner.start(prompt,workspace=workspace))
    assert forced.done and forced.truncated and forced.state.answer=='Paris'
    print(json.dumps(dict(passed=True,root_context_tokens=len(tokenizer.encode(serialized)),
                         terminal_submit=state.done,forced_submission_retained=True)))


if __name__=='__main__': asyncio.run(main())
