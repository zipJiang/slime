"""Native actor batches preserve missing questions as zero contributions."""
import copy
from runtime import EXPERIMENT
import sys
sys.path.append(str(EXPERIMENT/'snapshots/native-support-v1'))
from batches import training_data, partition_data


def split_mask_for_ranks(records, alignment):
    """Only if necessary, partition masks over repeated conditioning for DP sync.

    No new trainable token or fallback edge is introduced. Original edge/token
    denominators remain unchanged; auxiliary actor losses use the same disjoint
    masks. This is a sparse-batch fallback, not the usual packing path.
    """
    rows=copy.deepcopy(records)
    target=max(alignment, ((len(rows)+alignment-1)//alignment)*alignment)
    while len(rows)<target:
        index=max(range(len(rows)),key=lambda i:sum(rows[i]['loss_mask']))
        original=rows[index];positions=[i for i,m in enumerate(original['loss_mask']) if m]
        if len(positions)<2:
            raise ValueError('Not enough trainable tokens to distribute a sparse actor batch')
        left,right=copy.deepcopy(original),copy.deepcopy(original)
        midpoint=len(positions)//2
        for i in positions[midpoint:]:left['loss_mask'][i]=0
        for i in positions[:midpoint]:right['loss_mask'][i]=0
        for row in (left,right):row['metadata']['native_mask_partition']=True
        rows[index:index+1]=[left,right]
    return rows


def actor_data(records, expected_groups):
    expected=set(expected_groups);present={r['group_index'] for r in records}
    if not expected or not present<=expected:
        raise ValueError('Actor records contain an unexpected question')
    if not records:return None
    data=training_data(records,lane='actor',expected_groups=present)
    temperatures={'task':.6,'fold':.7}
    data['sampling_temperatures']=[temperatures[r['metadata']['tag']] for r in records]
    return data


def actor_packets(args,parallel,records,*,expected_groups):
    expected=set(expected_groups)
    if len(expected)!=args.global_batch_size:
        raise ValueError('Original question count must match the optimizer normalizer')
    data=actor_data(records,expected)
    if data is None:
        return None,dict(actor_step_skipped=True,reason='All actor edges filtered',
            empty_groups=sorted(expected),normalizing_questions=len(expected),mask_partitioned=False)
    present=set(data['rollout_ids']);packing=copy.copy(args);packing.global_batch_size=len(present)
    # Slime schedules only extant rows. Its loss reducer separately divides by
    # global_batch_sizes, which must still count the original questions.
    split=False
    try:packets=partition_data(packing,parallel,data)
    except AssertionError as exc:
        message=str(exc)
        if not ('samples < dp_size' in message or 'below the alignment threshold' in message):raise
        alignment=parallel['dp_size']*(parallel['microbatch_group_size_per_vp_stage'] if parallel['vpp_size']>1 else 1)
        records=split_mask_for_ranks(records,alignment)
        packets=partition_data(packing,parallel,actor_data(records,expected));split=True
    for packet in packets:packet['global_batch_sizes']=[len(expected)]
    return packets,dict(actor_step_skipped=False,empty_groups=sorted(expected-present),
        normalizing_questions=len(expected),mask_partitioned=split,records=len(records))
