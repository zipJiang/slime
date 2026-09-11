import json
from pathlib import Path
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from slime.ray.placement_group import InfoActor


def pinned_placement(args):
    training = args.actor_num_nodes * args.actor_num_gpus_per_node
    rollout = args.rollout_num_gpus
    bundles = ([{'CPU':1,'GPU':1,'deontic_direct_branch_train':0.001} for _ in range(training)] +
               [{'CPU':1,'GPU':1,'deontic_direct_branch_rollout':0.001} for _ in range(rollout)])
    pg = placement_group(bundles, strategy='PACK')
    ray.get(pg.ready(), timeout=180)
    actors = [InfoActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(len(bundles))]
    physical = ray.get([actor.get_ip_and_gpu_id.remote() for actor in actors])
    for actor in actors:
        ray.kill(actor)
    if len({ip for ip,gpu in physical[:training]}) != 1:
        raise ValueError('Training ranks must stay on one host')
    if {ip for ip,gpu in physical[:training]} & {ip for ip,gpu in physical[training:]}:
        raise ValueError('Training and rollout hosts must be disjoint')
    train_order = sorted(range(training), key=lambda i:int(physical[i][1]))
    from collections import Counter
    host_sizes = Counter(ip for ip, _ in physical[training:])
    # Slime's port allocator groups contiguous engines by num_gpus_per_node.
    # The new rollout pool uses three two-GPU hosts (num_gpus_per_node=2).
    rollout_order = sorted(range(training,len(bundles)), key=lambda i:(
        -host_sizes[physical[i][0]], physical[i][0], int(physical[i][1])))
    order = train_order+rollout_order
    ids = [physical[i][1] for i in order]
    output = Path(args.save).parent/'placement.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps([dict(role='train' if i<training else 'rollout',
        rank=i if i<training else i-training, host=physical[b][0],gpu=physical[b][1])
        for i,b in enumerate(order)],indent=2)+'\n')
    print(output.read_text(),flush=True)
    return {'actor':(pg,order,ids), 'critic':(pg,order,ids),
            'rollout':(pg,order[training:],ids[training:])}
