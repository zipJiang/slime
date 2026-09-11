"""Recovery must not drop a trained critic's moments or rewind question order."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from resume import role_arguments
from recipe import RECIPE_ID


class ResumeTests(unittest.TestCase):
    def fixture(self, root, iteration):
        (root/'recipe.json').write_text(json.dumps(dict(recipe_id=RECIPE_ID)))
        for role in ('actor', 'critic'):
            folder = root/role
            folder.mkdir()
            (folder/'latest_checkpointed_iteration.txt').write_text(str(iteration))
            count = max(0, iteration+1-6) if role == 'actor' else iteration+1
            (folder/f'iter_{iteration:07d}-readback.json').write_text(json.dumps(dict(
                checkpoint=str(folder/f'iter_{iteration:07d}'), role=role,
                expected_optimizer_steps=count, optimizer_steps=[count],
                full_storage_read=True, finite_tensors=True)))
        cursor = root/'actor/rollout'
        cursor.mkdir()
        (cursor/f'global_dataset_state_dict_{iteration}.pt').touch()
        return SimpleNamespace(load=str(root/'actor'), ppo_critic_load=str(root/'critic'),
            start_rollout_id=None, ckpt_step=None, finetune=False,
            num_critic_only_steps=6, rollout_global_dataset=True,
            no_load_optim=False, no_load_rng=True)

    def test_warmup_restores_trained_critic_and_cold_actor_separately(self):
        with TemporaryDirectory() as temp:
            args = self.fixture(Path(temp), 0)
            actor, critic, evidence = role_arguments(args)
            self.assertTrue(actor.no_load_optim)
            self.assertFalse(critic.no_load_optim)
            self.assertFalse(actor.no_load_rng or critic.no_load_rng)
            self.assertEqual(evidence['start_rollout_id'], 1)
            self.assertFalse(args.no_load_optim)  # Original recipe stays intact.

    def test_trained_actor_restores_both_optimizers(self):
        with TemporaryDirectory() as temp:
            args = self.fixture(Path(temp), 17)
            actor, critic, evidence = role_arguments(args)
            self.assertFalse(actor.no_load_optim or critic.no_load_optim)
            self.assertEqual(evidence['actor_updates'], 12)
            self.assertEqual(evidence['critic_updates'], 18)

    def test_rejects_incomplete_pairs_and_missing_cursor(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root, 5)
            tracker = root/'critic/latest_checkpointed_iteration.txt'
            tracker.write_text('4')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                role_arguments(args)
            tracker.write_text('5')
            (root/'actor/rollout/global_dataset_state_dict_5.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'question cursor'):
                role_arguments(args)

    def test_rejects_iteration_reset_and_cursor_override(self):
        with TemporaryDirectory() as temp:
            args = self.fixture(Path(temp), 5)
            args.finetune = True
            with self.assertRaisesRegex(ValueError, 'finetune'):
                role_arguments(args)
            args.finetune = False
            args.start_rollout_id = 0
            with self.assertRaisesRegex(ValueError, 'cursor'):
                role_arguments(args)

    def test_fresh_initialization_cannot_claim_completed_rounds(self):
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(load=temp, ppo_critic_load=None, start_rollout_id=1)
            with self.assertRaisesRegex(ValueError, 'round zero'):
                role_arguments(args)


if __name__ == '__main__':
    unittest.main()
