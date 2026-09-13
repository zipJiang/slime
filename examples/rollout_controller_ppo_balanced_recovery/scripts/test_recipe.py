"""Prevent posterior substitution and incompatible checkpoint continuation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from recipe import RECIPE_ID, estimator_for_round, target_source
from resume import role_arguments
from targets import checkpoint_fields
import test_resume
from test_targets import Tokenizer


class RecipeTests(unittest.TestCase):
    def test_warmup_boundary_is_explicit(self):
        self.assertEqual(estimator_for_round(9, 10), 'refined_td')
        self.assertEqual(estimator_for_round(10, 10), 'direct_branch_td')
        self.assertEqual(target_source('direct_branch_td'), 'direct_branch_mean')
        with self.assertRaises(ValueError):
            target_source('unknown')

    def test_critic_learns_mean_not_its_own_prior_blend(self):
        # A->B (.5), A->E (1), prior(A)=.5 and kappa=1:
        # critic target .75, actor baseline 2/3.
        record = dict(context='abc', target=.75, group_index=0,
            metadata=dict(lane='critic', node_id=0, value_version='v1',
                target_source='direct_branch_mean', diagnostics=dict(
                    mean_return=.75, refined_value=2/3, direct_branches=2)))
        kwargs = dict(sentinel_token_id=0, max_sequence_length=100)
        row = checkpoint_fields(record, Tokenizer(), **kwargs)
        self.assertEqual(row['reward'], .75)
        self.assertEqual(row['metadata']['target_source'], 'direct_branch_mean')
        with self.assertRaisesRegex(ValueError, 'Warm-up'):
            checkpoint_fields(record, Tokenizer(), warmup=True, **kwargs)
        record['target'] = 2/3
        with self.assertRaisesRegex(ValueError, 'unshrunk'):
            checkpoint_fields(record, Tokenizer(), **kwargs)

    def test_cannot_resume_original_ppo_into_new_recipe(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args = test_resume.ResumeTests().fixture(root, 17)
            role_arguments(args)
            (root/'recipe.json').write_text(json.dumps(dict(recipe_id='refined-td-original')))
            with self.assertRaisesRegex(ValueError, 'different estimator'):
                role_arguments(args)
            (root/'recipe.json').write_text(json.dumps(dict(recipe_id=RECIPE_ID)))
            self.assertEqual(role_arguments(args)[2]['actor_updates'], 8)


if __name__ == '__main__':
    unittest.main()
