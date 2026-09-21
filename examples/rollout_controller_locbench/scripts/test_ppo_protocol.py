import pytest
from ppo_protocol import schedule,lineage,validate_lineage,boundary,prefetch


def test_schedule_uses_every_training_question_without_heldout_or_duplicate_batch():
    split=dict(train=[str(i) for i in range(13)],development=['dev'],test=['test'])
    plan=schedule(split,updates=9,batch_size=4,warmup_ids=['0','1','2'])
    assert set(plan['questions'][:13])==set(split['train'])
    assert len(plan['questions'])==36 and all(len(set(batch))==4 for batch in plan['batches'])
    assert not set(plan['batches'][0])&{'0','1','2'}
    assert plan==schedule(split,updates=9,batch_size=4,warmup_ids=['0','1','2'])
    with pytest.raises(ValueError):schedule(dict(split,test=['0']))


def test_overlap_never_saves_an_untrained_prefetch_cursor():
    stamp=lineage(3,2,overlap=True)
    assert validate_lineage(stamp,3)==1
    with pytest.raises(ValueError):lineage(3,1,overlap=True)
    with pytest.raises(ValueError):lineage(3,2,overlap=False)
    for r in range(12):
        save=boundary(r,save_interval=4,last_round=11)
        collect=prefetch(r,last_round=11,save_now=save)
        assert not (save and collect)
    assert boundary(0,save_interval=4,last_round=11)
    assert boundary(1,save_interval=4,last_round=11)
