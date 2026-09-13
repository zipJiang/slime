import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from migration import checkpoint_boundary,migration_ready,release_idle_allocation


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        (self.run/'paused.json').write_text(json.dumps(dict(
            completed_actor_updates=32,completed_collection_rounds=42)))
        for role,count in [('actor',32),('critic',42)]:
            root=self.run/role;root.mkdir()
            (root/'latest_checkpointed_iteration.txt').write_text('41')
            (root/'iter_0000041-readback.json').write_text(json.dumps(dict(
                checkpoint=str(root/'iter_0000041'),role=role,expected_optimizer_steps=count,
                optimizer_steps=[count],full_storage_read=True,finite_tensors=True)))
        cursor=self.run/'actor/rollout';cursor.mkdir()
        (cursor/'global_dataset_state_dict_41.pt').touch()

    def test_complete_boundary(self):
        self.assertEqual(checkpoint_boundary(self.run)['actor_updates'],32)

    def test_paused_status_cannot_skip_an_unsaved_round(self):
        (self.run/'paused.json').write_text(json.dumps(dict(
            completed_actor_updates=33,completed_collection_rounds=43)))
        with self.assertRaises(ValueError):checkpoint_boundary(self.run)

    def test_audit_must_cover_current_role_and_checkpoint(self):
        path=self.run/'critic/iter_0000041-readback.json'
        original=json.loads(path.read_text())
        for key,value in [('finite_tensors',False),('full_storage_read',False),
                ('optimizer_steps',[41]),('checkpoint','/wrong'),('role','actor')]:
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(original,**{key:value})))
                with self.assertRaises(ValueError):checkpoint_boundary(self.run)

    def test_missing_cursor_is_rejected(self):
        (self.run/'actor/rollout/global_dataset_state_dict_41.pt').unlink()
        with self.assertRaises(ValueError):checkpoint_boundary(self.run)

    @patch('migration.subprocess.check_output')
    def test_waits_for_live_supervisor_and_transient_queries(self,query):
        query.return_value='123|RUNNING|\n'
        self.assertIsNone(migration_ready(123,self.run))
        query.side_effect=subprocess.TimeoutExpired('sacct',20)
        self.assertIsNone(migration_ready(123,self.run))

    @patch('migration.subprocess.check_output')
    def test_requires_successful_supervisor_exit(self,query):
        query.return_value='123|FAILED|\n'
        with self.assertRaises(RuntimeError):migration_ready(123,self.run)
        query.return_value='123|COMPLETED|\n'
        self.assertEqual(migration_ready(123,self.run)['iteration'],41)

    @patch('migration.subprocess.run')
    @patch('migration.subprocess.check_output')
    def test_never_releases_allocation_with_active_step(self,query,cancel):
        query.return_value='123.batch\n123.extern\n123.42\n'
        self.assertFalse(release_idle_allocation(123));cancel.assert_not_called()
        query.return_value='123.batch\n123.extern\n';cancel.return_value.returncode=0
        self.assertTrue(release_idle_allocation(123))
        self.assertEqual(cancel.call_args.args[0],['scancel','123'])


if __name__=='__main__':unittest.main()
