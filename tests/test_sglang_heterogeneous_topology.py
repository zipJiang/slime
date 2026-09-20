from argparse import Namespace


class _RemoteMethod:
    def __init__(self, actor):
        self.actor = actor

    def remote(self, start_port=10000, consecutive=1):
        self.actor.requests.append((start_port, consecutive))
        return self.actor.host, start_port


class _Actor:
    def __init__(self, host):
        self.host = host
        self.requests = []
        self._get_current_node_ip_and_free_port = _RemoteMethod(self)


def test_single_gpu_engine_ports_follow_actual_heterogeneous_hosts(monkeypatch):
    from slime.backends.sglang_utils import engine_group

    monkeypatch.setattr(engine_group.ray, "get", lambda value: value)
    actors = [_Actor("host-a") for _ in range(4)] + [_Actor("host-b") for _ in range(2)] + [
        _Actor("host-c") for _ in range(2)
    ]
    engines = list(enumerate(actors))

    ports, cursors = engine_group._allocate_rollout_engine_addr_and_ports_normal(
        args=Namespace(
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=4,
            sglang_dp_size=1,
        ),
        rollout_engines=engines,
        num_gpus_per_engine=1,
    )

    assert [ports[i]["host"] for i in range(8)] == [
        "host-a",
        "host-a",
        "host-a",
        "host-a",
        "host-b",
        "host-b",
        "host-c",
        "host-c",
    ]
    assert ports[0]["port"] == ports[4]["port"] == ports[6]["port"] == 15000
    assert ports[1]["port"] == ports[5]["port"] == ports[7]["port"] == 15033
    assert ports[2]["port"] == 15066
    assert ports[3]["port"] == 15099
    assert cursors == {"host-a": 15132, "host-b": 15066, "host-c": 15066}


def test_single_gpu_engine_rejects_host_migration(monkeypatch):
    from slime.backends.sglang_utils import engine_group

    actor = _Actor("host-a")
    calls = 0

    def get(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return value
        return "host-b", value[1]

    monkeypatch.setattr(engine_group.ray, "get", get)
    try:
        engine_group._allocate_rollout_engine_addr_and_ports_normal(
            args=Namespace(
                rollout_num_gpus_per_engine=1,
                num_gpus_per_node=4,
                sglang_dp_size=1,
            ),
            rollout_engines=[(0, actor)],
            num_gpus_per_engine=1,
        )
    except RuntimeError as exc:
        assert "moved hosts" in str(exc)
    else:
        raise AssertionError("host migration was not rejected")
