"""Versioned, fail-closed BrowserComp semantic judge for live tree terminals."""
import asyncio
import hashlib
import json

from examples.browsercomp_plus.judge import SYSTEM, TEMPLATE
from judge_contract import verdict


JUDGE_MODEL = 'Qwen/Qwen3.5-27B'
JUDGE_VERSION = 'browsecomp-plus/frozen-qwen3.5-27b/v1'


class SemanticJudge:
    """Score submitted answers through the same contract as trace collection.

    One instance belongs to one question.  Calls may run concurrently across tree
    expansions; the shared semaphore bounds the frozen judge service globally.
    """

    def __init__(self, client, url, question, references, *, semaphore=None,
                 attempts=3, retry_delay=2.0):
        if not question.strip() or not references or any(not str(r).strip() for r in references):
            raise ValueError('Semantic judge requires a question and reference answers')
        if attempts < 1 or retry_delay < 0:
            raise ValueError('Invalid semantic judge retry policy')
        self.client = client
        self.url = url.rstrip('/')
        self.question = question
        self.references = tuple(str(r) for r in references)
        self.semaphore = semaphore or asyncio.Semaphore(8)
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.records = []

    async def __call__(self, submitted):
        if not isinstance(submitted, str) or not submitted.strip():
            raise ValueError('Empty submissions are handled before semantic judging')
        prompt = TEMPLATE.format(question=self.question,
            reference=json.dumps(self.references), submitted=submitted)
        request = dict(model=JUDGE_MODEL,
            messages=[dict(role='system', content=SYSTEM), dict(role='user', content=prompt)],
            temperature=0, max_tokens=512,
            chat_template_kwargs=dict(enable_thinking=False),
            structured_outputs=dict(regex='(EQUIVALENT|DIFFERENT)'))
        async with self.semaphore:
            for attempt in range(1, self.attempts+1):
                try:
                    response = await self.client.post(self.url+'/chat/completions', json=request)
                    response.raise_for_status()
                    choice = response.json()['choices'][0]
                    text = choice['message']['content']
                    correct = verdict(text, choice['finish_reason'])
                    self.records.append(dict(version=JUDGE_VERSION, model=JUDGE_MODEL,
                        question_sha256=hashlib.sha256(self.question.encode()).hexdigest(),
                        references_sha256=hashlib.sha256(json.dumps(self.references).encode()).hexdigest(),
                        submitted=submitted, response=text,
                        finish_reason=choice['finish_reason'], correct=correct,
                        attempt=attempt))
                    return float(correct)
                except Exception:
                    if attempt == self.attempts:
                        raise
                    await asyncio.sleep(self.retry_delay)


def contract():
    return dict(version=JUDGE_VERSION, model=JUDGE_MODEL,
        prompt_sha256=hashlib.sha256((SYSTEM+TEMPLATE).encode()).hexdigest(),
        thinking=False, output_regex='(EQUIVALENT|DIFFERENT)')


__all__ = ['SemanticJudge', 'contract']
