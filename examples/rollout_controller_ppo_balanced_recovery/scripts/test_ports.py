"""Exercise native Slime port allocation on the actual ragged actor pool."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from placement import rollout_bundle_order


class Engine:
    def __init__(self, host):
        self._get_current_node_ip_and_free_port = SimpleNamespace(
            remote=lambda start_port=30000, consecutive=1: (host, start_port))


def allocator():
    source = Path(__file__).resolve().parents[1]/'snapshots/slime/slime/backends/sglang_utils/engine_group.py'
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
              n.name == '_allocate_rollout_engine_addr_and_ports_normal')
    namespace = dict(ray=SimpleNamespace(get=lambda x:x), logger=SimpleNamespace(info=lambda x:None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace[fn.name]


class PortTests(unittest.TestCase):
    def test_smaller_rollout_pools(self):
        for physical, count in [
            ([('t1','0'),('t1','1'),('t2','0'),('t2','1'),
              ('r1','0'),('r2','0'),('r1','1'),('r2','1')],3),
            ([('t','0'),('t','1'),('t','2'),('t','3'),('r','0'),('r','1')],1),
        ]:
            with self.subTest(count=count):
                self.check_ports(physical,count)

    def test_real_host_ports_do_not_collide_with_a_dedicated_critic(self):
        physical = [('train',str(i)) for i in range(4)] + [
            ('gh121','0'), ('gh121','1'), ('gh130','0'), ('gh203','0'), ('gh203','1'),
            ('gh130','1')]  # Last GPU serves the critic, not an actor engine.
        order = rollout_bundle_order(physical, 4, 5)
        self.assertEqual(order, [4,5,7,8,6])
        self.check_ports(physical,5)

    def check_ports(self, physical, count):
        order = rollout_bundle_order(physical,4,count)
        hosts = [physical[i][0] for i in order]
        ports, _ = allocator()(args=SimpleNamespace(num_gpus_per_node=2,
            rollout_num_gpus_per_engine=1, sglang_dp_size=1),
            rollout_engines=[(rank, Engine(host)) for rank,host in enumerate(hosts)])
        occupied = set()
        for rank, host in enumerate(hosts):
            self.assertEqual(ports[rank]['host'], host)
            for key in ('port','nccl_port','dist_init_addr'):
                port = ports[rank][key]
                if isinstance(port, str):
                    port = int(port.rsplit(':', 1)[1])
                self.assertNotIn((host,port), occupied)
                occupied.add((host,port))


if __name__ == '__main__':
    unittest.main()
