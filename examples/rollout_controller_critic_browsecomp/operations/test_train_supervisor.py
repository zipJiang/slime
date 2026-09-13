"""Exercise allocation routing without starting Slurm or Ray processes."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CRITIC_EXPERIMENT_ROOT', str(Path(__file__).resolve().parents[1]))
import train_supervisor as supervisor


class SupervisorTest(unittest.TestCase):
    def test_fresh_output_and_arbitrary_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'collection-run';base.mkdir()
            (base/'collection-finished.json').write_text('{}')
            (base/'supervisor.json').write_text('{"job": 123}')
            training=base/'training-new';operation=base/'operations-new'
            commands={};processes={}
            def start(name,command):
                commands[name]=command
                if name=='driver':
                    training.mkdir();(training/'complete.json').write_text('{}')
                proc=SimpleNamespace(returncode=0,wait=lambda **kw:0,
                    poll=lambda:None if name.startswith('ray-') else 0)
                processes[name]=proc
                return proc
            env=dict(CRITIC_TRAIN_JOBS='901:902',CRITIC_TRAIN_OUTPUT=str(training),
                CRITIC_TRAIN_OPERATIONS=str(operation),SLURM_JOB_ID='900',
                CRITIC_TRAIN_EXTRA_ARGS='["--lr", "1e-6"]')
            with patch.dict(os.environ,env), patch.object(supervisor.ops,'OUT',base), \
                 patch.object(supervisor.ops,'processes',processes), \
                 patch.object(supervisor.ops,'start',start), \
                 patch.object(supervisor,'job_active',return_value=False), \
                 patch.object(supervisor,'job_node',side_effect=lambda j:f'host{j}'), \
                 patch.object(supervisor,'allocated_gpus',return_value=2), \
                 patch.object(supervisor,'internal_ip',return_value='172.16.99.1'), \
                 patch.object(supervisor.subprocess,'run',return_value=SimpleNamespace(
                     returncode=0,stdout='4.0 GPU',stderr='')):
                supervisor.main()
            self.assertIn('--jobid=901',commands['ray-head'])
            self.assertIn('--jobid=902',commands['ray-worker-902'])
            self.assertIn('172.16.99.1:6475',commands['ray-worker-902'])
            self.assertIn('RAY_ADDRESS=172.16.99.1:6475',commands['driver'])
            self.assertIn(f'CRITIC_TRAIN_OUTPUT={training}',commands['driver'])
            self.assertEqual(commands['driver'][-2:],['--lr','1e-6'])
            self.assertTrue((operation/'complete.json').exists())
            self.assertFalse((base/'training').exists())
            self.assertEqual(json.loads((operation/'supervisor.json').read_text())['gpu_jobs'],[901,902])

    def test_existing_operations_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);operation=base/'ops';operation.mkdir()
            marker=operation/'failed.json';marker.write_text('preserved')
            with patch.dict(os.environ,{'CRITIC_TRAIN_OPERATIONS':str(operation)}), \
                 patch.object(supervisor.ops,'OUT',base), \
                 patch.object(supervisor,'operation_started',False), \
                 patch.object(supervisor,'job_node') as query:
                with self.assertRaises(FileExistsError): supervisor.main()
                query.assert_not_called()
                self.assertFalse(supervisor.operation_started)
            self.assertEqual(marker.read_text(),'preserved')


if __name__=='__main__':
    unittest.main()
