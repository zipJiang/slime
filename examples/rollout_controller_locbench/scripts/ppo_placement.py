"""Pin training, rollout, and portable-critic roles to dedicated pilot GPUs."""
from collections import Counter
import json
from pathlib import Path


def rollout_bundle_order(physical,training,rollout):
    host_sizes=Counter(ip for ip,_ in physical[training:training+rollout])
    return sorted(range(training,training+rollout),key=lambda i:(
        -host_sizes[physical[i][0]],physical[i][0],int(physical[i][1])))


def training_bundle_order(physical,training,tp_size=2):
    order=sorted(range(training),key=lambda i:(physical[i][0],int(physical[i][1])))
    if len(set((ip,int(gpu)) for ip,gpu in physical[:training]))!=training:
        raise ValueError('Training ranks repeat a physical GPU')
    for offset in range(0,training,tp_size):
        pair=order[offset:offset+tp_size]
        if len(pair)!=tp_size or len({physical[i][0] for i in pair})!=1:
            raise ValueError('Training tensor-parallel ranks must stay on one host')
    return order


def pinned_placement(args):
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from slime.ray.placement_group import InfoActor
    training=args.actor_num_nodes*args.actor_num_gpus_per_node
    rollout=args.rollout_num_gpus
    train_hosts=sorted(node['NodeManagerAddress'] for node in ray.nodes()
        if node['Alive'] and node['Resources'].get('locbench_ppo_train',0)>=args.actor_num_gpus_per_node)
    if len(train_hosts)!=args.actor_num_nodes:
        raise ValueError('Pilot training host count differs from configured topology')
    bundles=([{'CPU':1,'GPU':1,'locbench_ppo_train':.001,f'node:{host}':.001}
              for host in train_hosts for _ in range(args.actor_num_gpus_per_node)]+
        [{'CPU':1,'GPU':1,'locbench_ppo_rollout':.001} for _ in range(rollout)]+
        [{'CPU':1,'GPU':1,'locbench_ppo_replica':.001,
          f'node:{args.loc_critic_replica_host}':.001}])
    pg=placement_group(bundles,strategy='PACK');ray.get(pg.ready(),timeout=180)
    actors=[InfoActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,placement_group_bundle_index=i)).remote() for i in range(len(bundles))]
    physical=ray.get([actor.get_ip_and_gpu_id.remote() for actor in actors])
    for actor in actors: ray.kill(actor)
    if len({(ip,int(gpu)) for ip,gpu in physical})!=len(physical):
        raise ValueError('Training and inference repeat a physical GPU')
    train_order=training_bundle_order(physical,training,args.tensor_model_parallel_size)
    for offset in range(0,training,2):
        pair=[physical[i] for i in train_order[offset:offset+2]]
        if pair[0][0]=='172.16.203.27' and {int(g) for _,g in pair} not in ({0,1},{2,3}):
            raise ValueError('gh127 trainer TP pair does not follow its NVLink topology')
    rollout_order=rollout_bundle_order(physical,training,rollout)
    order=train_order+rollout_order;ids=[physical[i][1] for i in order]
    output=Path(args.save).parent/'placement.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps([
        dict(role='train' if i<training else 'rollout',rank=i if i<training else i-training,
             host=physical[b][0],gpu=physical[b][1]) for i,b in enumerate(order)]+
        [dict(role='critic_inference',host=physical[-1][0],gpu=physical[-1][1])],indent=2)+'\n')
    return dict(actor=(pg,order,ids),critic=(pg,order,ids),
        rollout=(pg,order[training:],ids[training:]),
        critic_inference=(pg,len(bundles)-1))
