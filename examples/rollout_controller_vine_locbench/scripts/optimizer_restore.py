"""Inspect restored native optimizer counters before a resumed PPO update."""


def inspect_optimizer(actor):
    wrappers = getattr(actor.optimizer, 'chained_optimizers', [actor.optimizer])
    steps = set()
    states = 0
    for wrapper in wrappers:
        optimizer = getattr(wrapper, 'optimizer', wrapper)
        states += len(optimizer.state)
        for group in optimizer.param_groups:
            if 'step' in group:
                steps.add(float(group['step']))
        for state in optimizer.state.values():
            if 'step' in state:
                steps.add(float(state['step']))
    return dict(steps=sorted(steps), states=states,
        scheduler_samples=float(actor.opt_param_scheduler.num_steps),
        no_load_optim=actor.args.no_load_optim, no_load_rng=actor.args.no_load_rng,
        finetune=actor.args.finetune)


def validate_restore(resume, reports, batch_size, world_size=4):
    for role in ('actor', 'critic'):
        expected = resume[f'{role}_updates']
        ranks = reports[role]
        if len(ranks) != world_size:
            raise ValueError(f'Missing {role} optimizer ranks')
        for rank, report in enumerate(ranks):
            if (report['steps'] != [expected] or report['states'] <= 0
                    or report['scheduler_samples'] != expected*batch_size
                    or report['no_load_optim'] or report['no_load_rng'] or report['finetune']):
                raise ValueError(f'{role} rank {rank} did not restore optimizer/RNG history: {report}')
    return dict(passed=True, resume=resume, world_size=world_size, reports=reports)
