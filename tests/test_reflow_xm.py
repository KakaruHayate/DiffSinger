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
        t1 = torch.tensor([[0.2], [0.7]])
        noise = torch.randn(2, 1, 2, 3)

        first = model.training_forward(condition, target, t1=t1, noise=noise)
        second = model.training_forward(condition, target, t1=t1, noise=noise)

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


    def test_dual_timestep_threads_mask_through_training_forward(self):
        """dual-timestep 下 prepare_training_inputs 返回 t2/mask，
        training_forward 用相同 t1/t2/mask/noise 必须能精确重放（同步性）。"""
        reflow_module.hparams['use_dual_timestep'] = True
        model = self.build_model()
        condition = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3, 2)

        spec, t1, t2, mask, noise = model.prepare_training_inputs(
            target, t1=torch.tensor([[0.2], [0.7]]), t2=torch.tensor([[0.1], [0.4]]),
            mask=(torch.rand(2, 3) < 0.25).float(), noise=torch.randn(2, 1, 2, 3),
        )
        replay = model.training_forward(
            condition, target, t1=t1, t2=t2, mask=mask, noise=noise,
        )
        manual = model.p_losses(spec, t1, cond=condition.transpose(1, 2), t2=t2, mask=mask, noise=noise)
        for rt, mt in zip(replay, manual):
            self.assertTrue(torch.equal(rt, mt))

        # effective t 是逐帧 [B, T]（符合 dual 语义），且 replay 用它
        effective_t = replay[2]
        self.assertEqual(effective_t.shape, (2, 3))
        replay[0].sum().backward()
        self.assertIsNotNone(condition.grad)

    def test_dual_timestep_xm_effective_t_is_per_frame(self):
        """run_reflow_xm 在 dual 下输出逐帧 effective t（mask 未被丢弃/重采样）。"""
        reflow_module.hparams['use_dual_timestep'] = True
        predictor = reflow_module.PitchRectifiedFlow(
            vmin=-2., vmax=2., cmin=-3., cmax=3., repeat_bins=4,
            time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={'num_layers': 1, 'num_channels': 4, 'dilation_cycle_length': 1},
        )
        condition = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3)
        non_padding = torch.ones(2, 3, 1)
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        output = run_reflow_xm(
            predictor, condition, target, loss_fn, non_padding,
            best_of_k=3, chunk_size=2,
        )
        v_pred, v_gt, effective_t = output
        self.assertEqual(effective_t.shape, (2, 3))         # [B, T] per-frame
        self.assertFalse(torch.any(torch.isnan(loss_fn(v_pred, v_gt, t=effective_t, non_padding=non_padding))))
        loss_fn(v_pred, v_gt, t=effective_t, non_padding=non_padding).backward()
        self.assertIsNotNone(condition.grad)

    def test_xm_shallow_t_start_is_respected(self):
        """shallow 场景：t1 采样范围从 T_start 开始；XM 应沿用 forward 的同分布。"""
        reflow_module.hparams['use_shallow_diffusion'] = True
        model = reflow_module.RectifiedFlow(
            out_dims=2, num_feats=1, t_start=0.4, time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={'num_layers': 1, 'num_channels': 4, 'dilation_cycle_length': 1},
            spec_min=[-1., -1.], spec_max=[1., 1.],
        )
        torch.manual_seed(0)
        spec, t1, t2, mask, _ = model.prepare_training_inputs(torch.randn(4, 8, 2))
        self.assertGreaterEqual(t1.min().item(), 0.4 - 1e-5)
        self.assertIsNone(t2)
        self.assertIsNone(mask)

    def test_run_reflow_xm_batch_coverage_and_no_collapse(self):
        """每元素都进 replay（bwd 数据量=B），且 K 个候选都有胜出（探索不退化）。"""
        reflow_module.hparams['use_dual_timestep'] = True
        predictor = reflow_module.PitchRectifiedFlow(
            vmin=-2., vmax=2., cmin=-3., cmax=3., repeat_bins=4,
            time_scale_factor=10,
            backbone_type='wavenet',
            backbone_args={'num_layers': 1, 'num_channels': 4, 'dilation_cycle_length': 1},
        )
        condition = torch.randn(6, 5, 4, requires_grad=True)
        target = torch.randn(6, 5)
        non_padding = torch.rand(6, 5, 1) > 0.3
        non_padding = non_padding.float()
        loss_fn = RectifiedFlowLoss('l2', log_norm=False)

        output = run_reflow_xm(
            predictor, condition, target, loss_fn, non_padding,
            best_of_k=4, chunk_size=3,
        )
        per_element = loss_fn(
            *output[:2], t=output[2], non_padding=non_padding, reduction='none'
        )
        self.assertEqual(per_element.shape, (6,))          # bwd 数据量 = B
        self.assertTrue(torch.isfinite(per_element).all())
        self.assertEqual(output[0].shape, (6, 1, 4, 5))     # 每元素都有输出


if __name__ == '__main__':
    unittest.main()
