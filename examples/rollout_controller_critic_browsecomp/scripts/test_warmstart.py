from types import SimpleNamespace
import pytest
from warmstart import initialization_cursor


def test_hf_and_native_model_only_initialization_both_start_fresh():
    args=SimpleNamespace(finetune=True,no_load_optim=True,no_load_rng=True)
    assert initialization_cursor(args,0)==0  # HF path: loaded iteration -1.
    assert initialization_cursor(args,1)==0  # Native finetune: loaded iteration 0.


def test_real_resume_retains_checkpoint_cursor():
    args=SimpleNamespace(finetune=False,no_load_optim=False,no_load_rng=False)
    assert initialization_cursor(args,16)==16


@pytest.mark.parametrize('flag',['no_load_optim','no_load_rng'])
def test_fresh_model_must_not_silently_restore_training_history(flag):
    args=SimpleNamespace(finetune=True,no_load_optim=True,no_load_rng=True)
    setattr(args,flag,False)
    with pytest.raises(ValueError,match='explicitly reset'):
        initialization_cursor(args,1)
