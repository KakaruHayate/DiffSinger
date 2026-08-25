import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import modules.core.reflow as reflow_module
from modules.core.xm import run_reflow_xm
from modules.losses.reflow_loss import RectifiedFlowLoss


class RectifiedFlowXMTest(unittest.TestCase):
    def setUp(self):
        self.original_hparams = dict(reflow_module.hparams)
        reflow_module.hparams.clear()
        reflow_module.hparams.update({
            'hidden_size': 4,
            'use_shallow_diffusion': False,
        })

    def tearDown(self):
        reflow_module.hparams.clear()
        reflow_module.hparams.update(self.original_hparams)

    def build_model(self):
        return reflow_module.RectifiedFlow(
            out_dims=2,
            num_feats=1,
            t_start=0.,
            time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={
                'num_layers': 1,
                'num_channels': 4,
                'dilation_cycle_length': 1,
            },
            spec_min=[-1., -1.],
            spec_max=[1., 1.],
        )

    def test_training_inputs_can_be_replayed_with_gradients(self):
        model = self.build_model()
        condition = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3, 2)
        timestep = torch.tensor([0.2, 0.7])
        noise = torch.randn(2, 1, 2, 3)

        first = model.training_forward(condition, target, t=timestep, noise=noise)
        second = model.training_forward(condition, target, t=timestep, noise=noise)

        for first_tensor, second_tensor in zip(first, second):
            self.assertTrue(torch.equal(first_tensor, second_tensor))
        first[0].sum().backward()
        self.assertIsNotNone(condition.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_per_sample_loss_ignores_padding(self):
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)
        prediction = torch.tensor(
            [[[[1., 2., 9.], [3., 4., 9.]]]],
            requires_grad=True,
        )
        target = torch.zeros_like(prediction)
        non_padding = torch.tensor([[[1.], [1.], [0.]]])

        loss = loss_fn(
            prediction,
            target,
            torch.tensor([0.5]),
            non_padding=non_padding,
            reduction='none',
        )

        self.assertTrue(torch.allclose(loss, torch.tensor([7.5])))
        loss.mean().backward()
        self.assertEqual(prediction.grad[0, 0, 0, 2].item(), 0.)
        self.assertEqual(prediction.grad[0, 0, 1, 2].item(), 0.)

    def test_mean_reduction_keeps_original_semantics(self):
        loss_fn = RectifiedFlowLoss('l1', log_norm=False)
        prediction = torch.tensor([[[[1., 2.]]], [[[3., 4.]]]])
        target = torch.zeros_like(prediction)
        timestep = torch.tensor([0.25, 0.75])

        original = loss_fn(prediction, target, timestep)
        expected = torch.nn.functional.l1_loss(prediction, target)

        self.assertTrue(torch.equal(original, expected))

    def test_pitch_predictor_supports_chunked_xm(self):
        predictor = reflow_module.PitchRectifiedFlow(
            vmin=-2.,
            vmax=2.,
            cmin=-3.,
            cmax=3.,
            repeat_bins=4,
            time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={
                'num_layers': 1,
                'num_channels': 4,
                'dilation_cycle_length': 1,
            },
        )
        condition = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3)
        non_padding = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        output = run_reflow_xm(
            predictor,
            condition,
            target,
            loss_fn,
            non_padding,
            best_of_k=3,
            chunk_size=2,
        )
        loss_fn(*output[:2], t=output[2], non_padding=non_padding).backward()

        self.assertEqual(output[0].shape, (2, 1, 4, 3))
        self.assertIsNotNone(condition.grad)

    def test_multi_variance_predictor_supports_list_targets(self):
        predictor = reflow_module.MultiVarianceRectifiedFlow(
            ranges=[(-2., 2.), (-1., 1.)],
            clamps=[(-2., 2.), (-1., 1.)],
            repeat_bins=2,
            time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={
                'num_layers': 1,
                'num_channels': 4,
                'dilation_cycle_length': 1,
            },
        )
        condition = torch.randn(2, 3, 4, requires_grad=True)
        targets = [torch.randn(2, 3), torch.randn(2, 3)]
        non_padding = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        output = run_reflow_xm(
            predictor,
            condition,
            targets,
            loss_fn,
            non_padding,
            best_of_k=3,
            chunk_size=2,
        )
        loss_fn(*output[:2], t=output[2], non_padding=non_padding).backward()

        self.assertEqual(output[0].shape, (2, 2, 2, 3))
        self.assertIsNotNone(condition.grad)


if __name__ == '__main__':
    unittest.main()
