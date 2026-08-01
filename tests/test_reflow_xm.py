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

        xm_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask)
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
        xm_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask)

        self.assertLess(xm_loss.item(), baseline.item())

    def test_best_bin_gradient_flows(self):
        predictor = self.build_pitch_predictor(repeat_bins=4)
        condition = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3)
        mask = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        v_pred, v_gt, t = predictor(condition, gt_spec=target, infer=False)
        loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask)
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
        xm_loss = loss_fn.forward_best_bin(v_pred, v_gt, t=t, non_padding=mask)
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


if __name__ == '__main__':
    unittest.main()
