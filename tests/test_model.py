"""
tests/test_model.py - Unit and tensor shape validation for MultiTaskMobileNetV3.
"""

import unittest
import torch
import torch.nn as nn

from core.model import MultiTaskMobileNetV3, MultiTaskCurrencyLoss, DEFAULT_DENOMINATIONS


class TestMultiTaskMobileNetV3(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = len(DEFAULT_DENOMINATIONS)
        self.model = MultiTaskMobileNetV3(
            num_denominations=self.num_classes,
            denominations=DEFAULT_DENOMINATIONS,
            pretrained=False
        ).to(self.device)
        self.model.eval()

    def test_single_image_forward_tensor_shapes(self):
        """Verify tensor outputs for a single image (B=1)."""
        x = torch.randn(1, 3, 224, 224, device=self.device)
        out = self.model(x)

        self.assertIn("denomination_logits", out)
        self.assertIn("counterfeit_logits", out)
        self.assertIn("counterfeit_prob", out)
        self.assertIn("features", out)

        self.assertEqual(out["denomination_logits"].shape, (1, self.num_classes))
        self.assertEqual(out["counterfeit_logits"].shape, (1, 1))
        self.assertEqual(out["counterfeit_prob"].shape, (1, 1))
        self.assertEqual(out["features"].shape, (1, 512))

        # Probability bounds
        prob = out["counterfeit_prob"].item()
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_batched_forward_tensor_shapes(self):
        """Verify tensor outputs for batch size > 1."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224, device=self.device)
        out = self.model(x)

        self.assertEqual(out["denomination_logits"].shape, (batch_size, self.num_classes))
        self.assertEqual(out["counterfeit_logits"].shape, (batch_size, 1))
        self.assertEqual(out["counterfeit_prob"].shape, (batch_size, 1))
        self.assertEqual(out["features"].shape, (batch_size, 512))

        probs = out["counterfeit_prob"].detach().cpu().numpy()
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_predict_single_helper(self):
        """Verify structured prediction output."""
        img = torch.randn(3, 224, 224, device=self.device)
        pred = self.model.predict_single(img)

        self.assertIn("denomination", pred)
        self.assertIn("denomination_idx", pred)
        self.assertIn("confidence", pred)
        self.assertIn("counterfeit_prob", pred)
        self.assertIn("is_counterfeit", pred)
        self.assertIn("verdict", pred)

        self.assertIsInstance(pred["denomination"], str)
        self.assertIsInstance(pred["confidence"], float)
        self.assertIsInstance(pred["is_counterfeit"], bool)
        self.assertIn(pred["verdict"], ["GENUINE", "SUSPECT / COUNTERFEIT"])

    def test_multi_task_loss_and_backward(self):
        """Verify joint loss computation and gradient backpropagation."""
        self.model.train()
        criterion = MultiTaskCurrencyLoss(alpha=1.0, beta=1.5)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        x = torch.randn(2, 3, 224, 224, device=self.device)
        target_denom = torch.tensor([1, 5], dtype=torch.long, device=self.device)
        target_counterfeit = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.device)

        optimizer.zero_grad()
        out = self.model(x)
        total_loss, l_denom, l_counterfeit = criterion(out, target_denom, target_counterfeit)

        self.assertGreater(total_loss.item(), 0.0)
        self.assertGreater(l_denom.item(), 0.0)
        self.assertGreater(l_counterfeit.item(), 0.0)

        total_loss.backward()
        # Verify gradient flow into shared fc and heads
        self.assertIsNotNone(self.model.shared_fc[0].weight.grad)
        self.assertIsNotNone(self.model.head_denomination[-1].weight.grad)
        self.assertIsNotNone(self.model.head_counterfeit[-1].weight.grad)
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
