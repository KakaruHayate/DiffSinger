import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import modules.core.reflow as reflow_module
from modules.losses.reflow_loss import RectifiedFlowLoss


class RectifiedFlowXMTest(unittest.TestCase):
    def setUp(self):
        self.original_hparams = dict(reflow_module.hparams)
        reflow_module.hparams.clear()
        reflow_module.hparams.update({
            'hidden_size': 4,
            'use_shallow_diffusion': False,
            'use_dual_timestep': False,
        })

    def tearDown(self):
        reflow_module.hparams.clear()
        reflow_module.hparams.update(self.original_hparams)

    def build_pitch_predictor(self, repeat_bins=8):
        return reflow_module.PitchRectifiedFlow(
            vmin=-2., vmax=2., cmin=-3., cmax=3.,
            repeat_bins=repeat_bins, time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={
                'num_layers': 1, 'num_channels': 4,
                'dilation_cycle_length': 1,
            },
        )

    def build_multi_variance_predictor(self, repeat_bins=4):
        return reflow_module.MultiVarianceRectifiedFlow(
            ranges=[(-2., 2.), (-1., 1.)],
            clamps=[(-2., 2.), (-1., 1.)],
            repeat_bins=repeat_bins, time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={
                'num_layers': 1, 'num_channels': 4,
                'dilation_cycle_length': 1,
            },
        )

    def test_best_bin_selects_minimum(self):
        predictor = self.build_pitch_predictor(repeat_bins=4)
        condition = torch.randn(1, 3, 4)
        target = torch.randn(1, 3)
        mask = torch.ones(1, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        torch.manual_seed(123)
        v_pred, v_gt, t = predictor(condition, gt_spec=target, infer=False)
        per_bin_loss = ((v_pred - v_gt) ** 2).mean(dim=-1)  # [1, 1, R]
        expected_min = per_bin_loss.min()

        xm_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=1)
        self.assertTrue(torch.allclose(xm_loss, expected_min, rtol=1e-5))

    def test_best_bin_lower_than_baseline(self):
        predictor = self.build_pitch_predictor(repeat_bins=8)
        condition = torch.randn(2, 3, 4)
        target = torch.randn(2, 3)
        mask = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        torch.manual_seed(42)
        v_pred, v_gt, t = predictor(condition, gt_spec=target, infer=False)
        baseline = loss_fn(v_pred, v_gt, t=t, non_padding=mask)
        xm_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=1)

        self.assertLess(xm_loss.item(), baseline.item())

    def test_best_bin_gradient_flows(self):
        predictor = self.build_pitch_predictor(repeat_bins=4)
        condition = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3)
        mask = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        v_pred, v_gt, t = predictor(condition, gt_spec=target, infer=False)
        loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=1)
        loss.backward()

        self.assertIsNotNone(condition.grad)
        self.assertTrue(any(
            p.grad is not None for p in predictor.parameters()
        ))

    def test_multi_variance_best_bin(self):
        predictor = self.build_multi_variance_predictor(repeat_bins=4)
        condition = torch.randn(2, 3, 4, requires_grad=True)
        targets = [torch.randn(2, 3), torch.randn(2, 3)]
        mask = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        v_pred, v_gt, t = predictor(condition, gt_spec=targets, infer=False)
        baseline = loss_fn(v_pred, v_gt, t=t, non_padding=mask)
        xm_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=1)
        xm_loss.backward()

        self.assertLess(xm_loss.item(), baseline.item())
        self.assertIsNotNone(condition.grad)

    def test_k1_preserves_original_semantics(self):
        loss_fn = RectifiedFlowLoss('l1', log_norm=False)
        prediction = torch.tensor([[[[1., 2.]]], [[[3., 4.]]]])
        target = torch.zeros_like(prediction)
        timestep = torch.tensor([0.25, 0.75])

        original = loss_fn(prediction, target, timestep)
        expected = torch.nn.functional.l1_loss(prediction, target)

        self.assertTrue(torch.equal(original, expected))

    def test_log_norm_with_non_uniform_padding(self):
        """Regression test for broadcasting bug in forward_best_bin."""
        predictor = self.build_pitch_predictor(repeat_bins=4)
        condition = torch.randn(2, 4, 4, requires_grad=True)
        target = torch.randn(2, 4)
        mask = torch.tensor([
            [[1.], [1.], [1.], [1.]],
            [[1.], [1.], [0.], [0.]],
        ])
        loss_fn = RectifiedFlowLoss('l2', log_norm=True)

        torch.manual_seed(999)
        v_pred, v_gt, t = predictor(condition, gt_spec=target, infer=False)
        xm_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=1)
        xm_loss.backward()

        masked_pred, masked_gt = loss_fn._mask_non_padding(v_pred, v_gt, mask)
        loss_all = loss_fn._forward(masked_pred, masked_gt, t=t)
        per_bin = loss_all.sum(dim=-1)
        mask_4d = mask.transpose(1, 2).unsqueeze(1).to(loss_all)
        mask_sum = mask_4d.sum(dim=-1)
        per_bin = per_bin / mask_sum.expand_as(per_bin).clamp_min(1.)
        topk_values, topk_indices = per_bin.topk(1, dim=-1, largest=False)
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, v_gt.shape[-1])
        best_pred = masked_pred.gather(2, gather_indices)
        best_gt = masked_gt.gather(2, gather_indices)
        best_loss = loss_fn.loss(best_pred, best_gt)
        best_loss = loss_fn.get_weights(t).squeeze(2).unsqueeze(2) * best_loss
        best_loss = best_loss * mask.transpose(1, 2).unsqueeze(2)
        expected = best_loss.mean()

        self.assertTrue(torch.allclose(xm_loss, expected, rtol=1e-5))
        self.assertIsNotNone(condition.grad)

    def test_best_half_bins_lower_than_best_one(self):
        """Selecting k=R/2 bins should be better than k=1 but worse than mean."""
        predictor = self.build_pitch_predictor(repeat_bins=8)
        condition = torch.randn(2, 3, 4)
        target = torch.randn(2, 3)
        mask = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        torch.manual_seed(77)
        v_pred, v_gt, t = predictor(condition, gt_spec=target, infer=False)
        baseline = loss_fn(v_pred, v_gt, t=t, non_padding=mask)
        k1_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=1)
        k4_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=4)

        self.assertLess(k1_loss.item(), k4_loss.item())
        self.assertLess(k4_loss.item(), baseline.item())

    def test_multi_variance_k_half_preserves_all_features(self):
        """With F=2 and k=R/2, both features must receive gradients."""
        predictor = self.build_multi_variance_predictor(repeat_bins=8)
        condition = torch.randn(2, 3, 4, requires_grad=True)
        targets = [torch.randn(2, 3), torch.randn(2, 3)]
        mask = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        v_pred, v_gt, t = predictor(condition, gt_spec=targets, infer=False)
        loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask, k=4)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(condition.grad)


if __name__ == '__main__':
    unittest.main()
