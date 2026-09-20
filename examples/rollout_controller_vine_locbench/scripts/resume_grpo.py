"""Require a fully audited GRPO actor checkpoint and matching dataset cursor."""
import json
from pathlib import Path

from grpo_recipe import RECIPE_ID, RESUME_COMPATIBLE_RECIPE_IDS


def prepare_resume(args):
    root = Path(args.load)
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        if args.start_rollout_id not in (None, 0):
            raise ValueError("Fresh HF initialization starts at round zero")
        return None
    recipe = json.loads((root.parent / "recipe.json").read_text())
    source_recipe_id = recipe.get("recipe_id")
    if source_recipe_id not in RESUME_COMPATIBLE_RECIPE_IDS:
        raise ValueError("Resume recipe is not explicitly compatible with this GRPO recipe")
    if (root.parent / "scientific-rejection.json").exists():
        raise ValueError("Cannot resume a scientifically rejected run")
    iteration = int(tracker.read_text().strip())
    if iteration < 0:
        raise ValueError("Invalid checkpoint iteration")
    start = iteration + 1
    if args.start_rollout_id not in (None, start) or args.ckpt_step not in (None, iteration):
        raise ValueError("Resume cursor disagrees with checkpoint")
    if args.finetune or args.no_load_optim or args.no_load_rng:
        raise ValueError("Resume must retain optimizer, scheduler, and RNG history")
    checkpoint = root / f"iter_{iteration:07d}"
    audit = json.loads((root / f"iter_{iteration:07d}-readback.json").read_text())
    if (
        Path(audit["checkpoint"]).resolve() != checkpoint.resolve()
        or audit["role"] != "actor"
        or audit["expected_optimizer_steps"] != start
        or audit["optimizer_steps"] != [start]
        or not audit["full_storage_read"]
        or not audit["finite_tensors"]
    ):
        raise ValueError("Checkpoint requires a matching full tensor readback audit")
    cursor = root / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
    if not args.rollout_global_dataset or not cursor.is_file():
        raise ValueError("Resume requires the saved question cursor")
    saved = json.loads(
        (root.parent / "checkpoints" / f"round-{iteration:04d}.json").read_text()
    )
    if saved["actor_updates"] != start or saved["round_id"] != iteration:
        raise ValueError("Checkpoint was not committed at the matching training boundary")
    previous = recipe["arguments"]
    for key in (
        "global_batch_size",
        "rollout_batch_size",
        "num_steps_per_rollout",
        "lr",
        "kl_loss_coef",
        "rollout_shuffle",
        "tensor_model_parallel_size",
        "grpo_group_size",
    ):
        if previous[key] != getattr(args, key):
            raise ValueError(f"Resume changes training setting: {key}")
    if Path(previous["prompt_data"]).read_bytes() != Path(args.prompt_data).read_bytes():
        raise ValueError("Resume changes training schedule")
    args.start_rollout_id = start
    return {
        "iteration": iteration,
        "start_rollout_id": start,
        "actor_updates": start,
        "question_cursor": str(cursor.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "source_recipe_id": source_recipe_id,
        "current_recipe_id": RECIPE_ID,
        "recipe_change": (
            None
            if source_recipe_id == RECIPE_ID
            else {"previous": source_recipe_id, "current": RECIPE_ID}
        ),
    }


def validate_actor_restore(resume, reports, batch_size, world_size):
    expected = resume["actor_updates"]
    if len(reports) != world_size:
        raise ValueError("Missing actor optimizer ranks")
    for rank, report in enumerate(reports):
        if (
            report["steps"] != [expected]
            or report["states"] <= 0
            or report["scheduler_samples"] != expected * batch_size
            or report["no_load_optim"]
            or report["no_load_rng"]
            or report["finetune"]
        ):
            raise ValueError(f"Actor rank {rank} failed optimizer restoration: {report}")
    return {
        "passed": True,
        "resume": resume,
        "world_size": world_size,
        "reports": reports,
    }
