import unittest

import torch
from torch import nn

from training.weight_averaging import ExponentialMovingAverage


class ExponentialMovingAverageTest(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
            self.model.bias.fill_(2.0)
        self.ema = ExponentialMovingAverage(
            dict(self.model.named_parameters()), decay=0.75
        )

    def test_step_uses_exponential_average(self):
        with torch.no_grad():
            self.model.weight.fill_(5.0)
            self.model.bias.fill_(6.0)
        self.ema.step()
        torch.testing.assert_close(
            self.ema.shadow['weight'], torch.full_like(self.model.weight, 2.0)
        )
        torch.testing.assert_close(
            self.ema.shadow['bias'], torch.full_like(self.model.bias, 3.0)
        )

    def test_apply_and_restore_preserve_parameter_objects(self):
        weight_id = id(self.model.weight)
        with torch.no_grad():
            self.model.weight.fill_(5.0)
        self.ema.apply()
        self.assertEqual(id(self.model.weight), weight_id)
        torch.testing.assert_close(
            self.model.weight, torch.full_like(self.model.weight, 1.0)
        )
        self.ema.restore()
        self.assertEqual(id(self.model.weight), weight_id)
        torch.testing.assert_close(
            self.model.weight, torch.full_like(self.model.weight, 5.0)
        )

    def test_state_dict_round_trip(self):
        with torch.no_grad():
            self.model.weight.fill_(5.0)
        self.ema.step()
        state_dict = self.ema.state_dict()
        restored = ExponentialMovingAverage(
            dict(self.model.named_parameters()), decay=0.75
        )
        restored.load_state_dict(state_dict)
        for name in state_dict:
            torch.testing.assert_close(restored.shadow[name], state_dict[name])

    def test_strict_load_rejects_missing_keys(self):
        with self.assertRaises(KeyError):
            self.ema.load_state_dict({'weight': self.model.weight.detach().clone()})

    def test_empty_parameter_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            ExponentialMovingAverage({}, decay=0.999)

    def test_nested_apply_is_rejected(self):
        self.ema.apply()
        try:
            with self.assertRaises(RuntimeError):
                self.ema.apply()
        finally:
            self.ema.restore()


if __name__ == '__main__':
    unittest.main()
