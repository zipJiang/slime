"""Evaluate one frozen LocBench actor checkpoint on the repository-disjoint test set."""
import argparse
import asyncio
import json
import time
from pathlib import Path

from synthetic_data import (
    CACHE,
    E,
    MODEL,
    LoggedPolicy,
    PolicyFormat,
    SamplingParams,
    digest,
    load_cases,
    metrics,
    prepare,
    save,
    score,
    write,
)
from synthetic_runtime import environment, make_world
from runtime_v2 import verify_harness


def read(path):
    return json.loads(Path(path).read_text())


def reconcile_manifest(path, requested):
    """Allow retry-budget increases without changing the evaluation protocol."""
    if not path.exists():
        write(path, requested)
        return requested
    existing = read(path)
    existing_stable = {key: value for key, value in existing.items() if key != 'attempts'}
    requested_stable = {key: value for key, value in requested.items() if key != 'attempts'}
    if existing_stable != requested_stable:
        raise ValueError('Checkpoint evaluation manifest changed')
    if requested['attempts'] < existing['attempts']:
        raise ValueError('Checkpoint evaluation retry allowance cannot decrease')
    if requested['attempts'] > existing['attempts']:
        write(path, requested)
        return requested
    return existing


async def main(args):
    from transformers import AutoTokenizer

    verify_harness()
    args.output.mkdir(parents=True, exist_ok=True)
    infra = read(args.infra)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    policy_format = PolicyFormat.resolve(MODEL, tokenizer=tokenizer, profile='qwen_xml')
    cases = load_cases(E / 'data/test.jsonl')
    case_ids = [case.id for case in cases]
    if len(case_ids) != 87 or len(case_ids) != len(set(case_ids)):
        raise ValueError('Expected the frozen 87-question LocBench test split')
    manifest = dict(
        protocol='locbench-improved-ppo-checkpoint-test-v1',
        checkpoint=args.checkpoint,
        model=str(args.model.resolve()),
        questions=case_ids,
        samples=1,
        attempts=args.attempts,
        concurrency_per_server=args.concurrency_per_server,
        environment=environment(),
        seed_namespace='locbench-improved-ppo-checkpoint-test-v1',
        split_sha256=digest(E / 'data/split.json'),
        test_sha256=digest(E / 'data/test.jsonl'),
        source_sha256=digest(__file__),
        scope='All 87 repository-disjoint test questions; one paired sample per checkpoint',
    )
    manifest_path = args.output / 'manifest.json'
    reconcile_manifest(manifest_path, manifest)
    urls = infra['actor_urls']
    if not urls:
        raise ValueError('No actor inference URL')
    semaphores = [asyncio.Semaphore(args.concurrency_per_server) for _ in urls]

    async def one(case, index):
        case_root = args.output / 'cases' / case.id
        for attempt in range(args.attempts):
            out = case_root / f'attempt-{attempt}'
            complete = out / 'complete.json'
            if complete.exists():
                return read(complete)
            url_index = index % len(urls)
            async with semaphores[url_index]:
                policy = LoggedPolicy(
                    output=out,
                    key=f'locbench-improved-ppo-checkpoint-test-v1/{case.id}/{attempt}',
                    model=MODEL,
                    served_model='locbench-qwen35-9b',
                    format=policy_format,
                    base_url=urls[url_index],
                    api_key='EMPTY',
                    timeout=1800,
                    version=args.checkpoint,
                    default_params=SamplingParams(
                        max_tokens=8192, temperature=.6, top_p=.95, top_k=20, logprobs=1),
                )
                state = None
                started = time.monotonic()
                try:
                    repo = await asyncio.to_thread(prepare, CACHE, case.repo, case.base_commit)
                    runner, workspace, prompt, _ = make_world(case, repo, policy)
                    runner._sampling_params = SamplingParams(
                        max_tokens=8192, temperature=.6, top_p=.95, top_k=20)
                    state = await runner.start(prompt, workspace=workspace)
                    while not state.done and not state.truncated:
                        before = len(state.turns)
                        state = await runner.advance(state)
                        if len(state.turns) == before:
                            raise ValueError('No rollout progress')
                        write(out / 'progress.json', dict(
                            turns=state.turns_taken, folds=state.folds, time=time.time()))
                        save(out / 'partial.pkl.gz', state.snapshot())
                    if not state.done:
                        state = await runner.finish(state)
                    detail = metrics(state, policy)
                    result = dict(
                        query_id=case.id,
                        checkpoint=args.checkpoint,
                        attempt=attempt,
                        reward=score(state.state.locations, case.gold)['reward'],
                        submitted=state.state.submitted,
                        horizon_hit=state.turns_taken + state.folds >= 80,
                        seconds=time.monotonic() - started,
                        **{key: value for key, value in detail.items() if key != 'events'},
                    )
                    save(out / 'trace.pkl.gz', state.snapshot())
                    write(out / 'events.json', detail['events'])
                    write(complete, result)
                    print(json.dumps({key: result[key] for key in (
                        'query_id', 'checkpoint', 'reward', 'total_turns', 'folds',
                        'repeated_calls', 'seconds')}), flush=True)
                    return result
                except Exception as exc:
                    write(out / 'failure.json', dict(
                        query_id=case.id, checkpoint=args.checkpoint,
                        attempt=attempt, error=repr(exc), time=time.time()))
                    if state is not None:
                        save(out / 'partial.pkl.gz', state.snapshot())
                finally:
                    await policy.aclose()
        return dict(query_id=case.id, checkpoint=args.checkpoint,
                    error=f'Failed after {args.attempts} attempts')

    results = await asyncio.gather(*(one(case, index) for index, case in enumerate(cases)))
    failures = [row for row in results if 'error' in row]
    rewards = [row['reward'] for row in results if 'error' not in row]
    write(args.output / 'results.json', results)
    summary = dict(
        checkpoint=args.checkpoint,
        model=str(args.model.resolve()),
        questions=len(results),
        completed=len(rewards),
        failures=len(failures),
        mean_reward=sum(rewards) / len(rewards) if rewards else None,
        exact_successes=sum(reward == 1 for reward in rewards),
        exact_success_rate=(sum(reward == 1 for reward in rewards) / len(rewards)) if rewards else None,
        time=time.time(),
    )
    write(args.output / 'summary.json', summary)
    if failures:
        raise RuntimeError(f'{len(failures)} test trajectories failed')
    write(args.output / 'complete.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infra', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--concurrency-per-server', type=int, default=8)
    args = parser.parse_args()
    if args.attempts < 1 or args.concurrency_per_server < 1:
        raise ValueError('Attempts and concurrency must be positive')
    asyncio.run(main(args))
