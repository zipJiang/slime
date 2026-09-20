"""Guard the flag that OOMed LocBench on Sep 17.

`train_vine.py` sets `args.vine_chunked_logits = True`, but the pinned Slime
snapshot reads it with `getattr(..., False)`. When the snapshot does not
implement the flag the setting is silently inert: full-vocabulary logits keep
their FP32 upcast, and training OOMs on the first step that runs with resident
Adam moments (round 1, not round 0). These tests fail loudly on that state
rather than waiting for the crash.

See operations/locbench-oom-sep17/failure.json.
"""
from pathlib import Path
import re

EXPERIMENT = Path(__file__).resolve().parents[1]
SNAPSHOT = EXPERIMENT/'snapshots/slime'
MODEL = SNAPSHOT/'slime/backends/megatron_utils/model.py'
LOSS = SNAPSHOT/'slime/backends/megatron_utils/loss.py'
FLAG = 'vine_chunked_logits'


def test_train_vine_enables_the_flag():
    assert f'args.{FLAG} = True' in (EXPERIMENT/'scripts/train_vine.py').read_text()


def test_run_ppo_supplies_the_preconditions():
    text = (EXPERIMENT/'scripts/run_ppo.sh').read_text()
    assert '--log-probs-chunk-size 1024' in text
    assert '--rollout-temperature 1' in text


def test_snapshot_actually_implements_the_flag():
    # The defect: the launcher sets the flag and the snapshot never reads it.
    assert FLAG in MODEL.read_text(), f'{MODEL} does not honour {FLAG}'
    assert FLAG in LOSS.read_text(), f'{LOSS} does not honour {FLAG}'


def test_every_training_forward_disables_fp32_output_under_the_flag():
    text = MODEL.read_text()
    calls = [m.start() for m in re.finditer(r'output_tensor = model\(\*\*forward_kwargs\)', text)]
    assert len(calls) == 2, f'expected 2 training forwards, found {len(calls)}'
    for start in calls:
        window = text[max(0, start-260):start]
        assert FLAG in window and "forward_kwargs['fp32_output'] = False" in window, (
            'a training forward still upcasts full-vocabulary logits to FP32')


def test_loss_accepts_bf16_only_behind_the_flag():
    text = LOSS.read_text()
    assert 'torch.float32, torch.bfloat16' in text
    # Two strict asserts survive: the chunk-level kernel contract, and the
    # else-branch taken whenever the flag is off.
    assert text.count('assert logits.dtype == torch.float32') == 2
    guarded = text.split('assert non_loss_data', 1)[1][:400]
    assert 'else:' in guarded and 'assert logits.dtype == torch.float32' in guarded


def test_chunk_level_kernel_assert_is_untouched():
    # The `getattr` gate must not have leaked into the chunk-level function,
    # which is always handed FP32 by the chunking path itself.
    head = LOSS.read_text().split('assert non_loss_data', 1)[0]
    assert head.count('assert logits.dtype == torch.float32') == 1
    assert FLAG not in head


def test_loss_guards_temperature_and_chunk_size():
    text = LOSS.read_text()
    assert 'rollout_temperature != 1.0' in text and 'log_probs_chunk_size <= 0' in text


def test_chunked_kernel_is_unchanged_from_the_pinned_snapshot():
    # ppo_utils.py carries the kernel itself and must not drift with this fix.
    original = (EXPERIMENT.parent/'rollout_controller_locbench/snapshots/slime'
                /'slime/utils/ppo_utils.py').read_bytes()
    assert (SNAPSHOT/'slime/utils/ppo_utils.py').read_bytes() == original


def test_locbench_specific_pins_survive_the_copy():
    original_root = EXPERIMENT.parent/'rollout_controller_locbench/snapshots/slime'
    for relative in ('slime/backends/megatron_utils/checkpoint.py',
                     'slime/backends/sglang_utils/engine_group.py'):
        assert (SNAPSHOT/relative).read_bytes() == (original_root/relative).read_bytes(), relative
