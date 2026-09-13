"""Keep fresh critic initialization independent from the pretraining cursor."""


def initialization_cursor(args, native_cursor):
    # Megatron finetune loads model weights and returns iteration 0. Slime then
    # adds one, as it would for a real resumed checkpoint. Fresh HF loading uses
    # iteration -1 instead. Both fresh paths must expose rollout zero to PPO.
    if args.finetune:
        if not args.no_load_optim or not args.no_load_rng:
            raise ValueError('Fresh critic initialization must explicitly reset optimizer and RNG')
        return 0
    return native_cursor
