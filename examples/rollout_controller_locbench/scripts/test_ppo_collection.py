from types import SimpleNamespace
import pytest
from ppo_collection import command,router_address


def test_collector_uses_manager_endpoint_and_rejects_unset_driver_port():
    args=SimpleNamespace(sglang_router_ip=None,sglang_router_port=None,
        rollout_batch_size=4,loc_max_pass_attempts=32,loc_search_concurrency=4,loc_prior_strength=1.)
    frozen=dict(value_url='http://critic:1234',policy_version='actor-0000',
        server_weight_version='1',value_version='critic-0000',seed_namespace='test')
    options=dict(questions='questions.json',output='batch',frozen=frozen,cutoff=None,pass_tokens=32768)
    with pytest.raises(ValueError,match='endpoint'):command(args,**options)
    # Mimic Ray's isolated argument copy and deployment-assigned port.
    manager=SimpleNamespace(args=SimpleNamespace(sglang_router_ip='172.16.203.27',sglang_router_port=4217))
    endpoint=router_address(manager)
    assert args.sglang_router_port is None
    args.sglang_router_ip=endpoint['host'];args.sglang_router_port=endpoint['port']
    argv=command(args,**options)
    assert argv[argv.index('--url')+1]=='http://172.16.203.27:4217'
