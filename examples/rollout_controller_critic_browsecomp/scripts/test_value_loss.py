import unittest
import torch
from value_loss import regression_errors


class ValueLossTests(unittest.TestCase):
    def test_warmup_moves_probability_toward_success(self):
        logit = torch.tensor([0.], requires_grad=True)
        loss, _, _ = regression_errors(logit.sigmoid(), torch.ones(1), torch.tensor([.5]), warmup=True, clip=.2)
        loss.sum().backward()
        self.assertLess(logit.grad.item(), 0)

    def test_joint_clipping_uses_frozen_probability(self):
        loss, clipped, mse = regression_errors(torch.tensor([.9]), torch.ones(1),
                                               torch.tensor([.5]), warmup=False, clip=.2)
        self.assertAlmostEqual(loss.item(), .09, places=6)
        self.assertAlmostEqual(mse.item(), .01, places=6)
        self.assertEqual(clipped.item(), 1)

    def test_invalid_return_is_rejected(self):
        with self.assertRaises(ValueError):
            regression_errors(torch.tensor([.5]), torch.tensor([1.1]), torch.tensor([.5]), warmup=True, clip=.2)


if __name__ == '__main__':
    unittest.main()
