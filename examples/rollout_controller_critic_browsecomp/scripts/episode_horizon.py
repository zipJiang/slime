"""One task-turn horizon shared by root rollouts and every search fork."""


async def iter_episode(runner, state, *, task_limit=48, sampling_params=None):
    state.check_continuation()
    if state.done or state.truncated:
        yield state.snapshot()
        return
    while state.turns_taken < task_limit:
        before = len(state.turns)
        state = await runner.advance(state, sampling_params=sampling_params)
        if state.done or state.truncated or len(state.turns) == before:
            break
        if state.turns[-1].tag == 'fold':
            yield state.snapshot()
    if not state.done and not state.truncated:
        # This one forced submission is additional to the ordinary task turns,
        # exactly as in root evaluation. It remains recorded and trainable.
        state = await runner.finish(state, sampling_params=sampling_params)
    if state.turns_taken > task_limit + 1:
        raise ValueError('Rollout exceeded its shared task horizon')
    yield state.snapshot()


class _EpisodeDriver:
    def __init__(self, runner, task_limit):
        self.runner = runner
        self.task_limit = task_limit

    def iter_run(self, state, *, sampling_params=None):
        return iter_episode(self.runner, state, task_limit=self.task_limit,
                            sampling_params=sampling_params)


def with_episode_horizon(expander, task_limit):
    # Keep the pinned native expander's proposal draw, fork isolation, edge
    # provenance, and fold checkpoints. Only its episode stopping rule changes.
    expander._runner = _EpisodeDriver(expander._runner, task_limit)
    return expander
