import asyncio

import pytest

from semantic_judge import JUDGE_MODEL, SemanticJudge, contract


class Response:
    def __init__(self, payload, *, error=None):
        self.payload, self.error = payload, error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def post(self, url, json):
        self.requests.append((url, json))
        return next(self.responses)


def answer(text='EQUIVALENT', finish='stop'):
    return Response({'choices': [{'message': {'content': text}, 'finish_reason': finish}]})


@pytest.mark.asyncio
async def test_equivalent_answer_records_complete_versioned_evidence():
    client = Client([answer()])
    judge = SemanticJudge(client, 'http://judge/v1/', 'Where?', ['Paris'], retry_delay=0)
    assert await judge('Paris, France') == 1.0
    url, request = client.requests[0]
    assert url == 'http://judge/v1/chat/completions'
    assert request['model'] == JUDGE_MODEL
    assert request['temperature'] == 0
    assert request['chat_template_kwargs'] == {'enable_thinking': False}
    assert judge.records[0]['submitted'] == 'Paris, France'
    assert judge.records[0]['correct'] is True
    assert len(contract()['prompt_sha256']) == 64


@pytest.mark.asyncio
async def test_difference_is_a_valid_zero_outcome():
    judge = SemanticJudge(Client([answer('DIFFERENT')]), 'http://judge',
                          'Where?', ['Paris'], retry_delay=0)
    assert await judge('London') == 0.0


@pytest.mark.asyncio
async def test_transport_failure_retries_then_succeeds():
    client = Client([Response({}, error=RuntimeError('temporary')), answer()])
    judge = SemanticJudge(client, 'http://judge', 'Where?', ['Paris'], attempts=2,
                          retry_delay=0)
    assert await judge('Paris') == 1.0
    assert len(client.requests) == 2
    assert judge.records[0]['attempt'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [answer('maybe'), answer('EQUIVALENT', 'length')])
async def test_malformed_or_incomplete_output_fails_closed(response):
    judge = SemanticJudge(Client([response]), 'http://judge', 'Where?', ['Paris'],
                          attempts=1, retry_delay=0)
    with pytest.raises(ValueError):
        await judge('Paris')
    assert judge.records == []


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        SemanticJudge(Client([]), 'http://judge', '', ['Paris'])
    with pytest.raises(ValueError):
        SemanticJudge(Client([]), 'http://judge', 'Where?', [])
    with pytest.raises(ValueError):
        SemanticJudge(Client([]), 'http://judge', 'Where?', ['Paris'], attempts=0)
