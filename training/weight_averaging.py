from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class ExponentialMovingAverage:
    """Maintain an exponential moving average of trainable parameters.

    The shadow tensors are checkpointed separately from the model parameters.
    ``apply`` and ``restore`` temporarily swap values in-place so optimizer
    parameter references remain valid.
    """

    def __init__(self, parameters: Mapping[str, nn.Parameter], decay: float = 0.999):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"EMA decay must be in (0, 1], got {decay}.")
        self.decay = float(decay)
        self.referenced = {
            name: parameter
            for name, parameter in parameters.items()
            if parameter.requires_grad
        }
        if not self.referenced:
            raise ValueError("EMA did not match any trainable parameters.")
        self.shadow: dict[str, Tensor] = {}
        self.backup: dict[str, Tensor] = {}
        self.register()

    def __len__(self) -> int:
        return len(self.shadow)

    @property
    def applied(self) -> bool:
        return bool(self.backup)

    @torch.no_grad()
    def register(self) -> None:
        """Reset shadow values to the parameters currently referenced."""
        if self.applied:
            raise RuntimeError("Cannot register EMA while shadow parameters are applied.")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in self.referenced.items()
        }

    @torch.no_grad()
    def step(self) -> None:
        """Update shadow values after a real optimizer step."""
        if self.applied:
            raise RuntimeError("Cannot update EMA while shadow parameters are applied.")
        one_minus_decay = 1.0 - self.decay
        for name, parameter in self.referenced.items():
            shadow = self.shadow[name]
            if shadow.device != parameter.device:
                shadow = shadow.to(parameter.device)
                self.shadow[name] = shadow
            source = parameter.detach().to(dtype=shadow.dtype)
            shadow.mul_(self.decay).add_(source, alpha=one_minus_decay)

    @torch.no_grad()
    def apply(self) -> None:
        """Temporarily replace referenced parameters with their EMA values."""
        if self.applied:
            raise RuntimeError("EMA parameters are already applied.")
        for name, parameter in self.referenced.items():
            self.backup[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        """Restore parameters saved by the matching ``apply`` call."""
        if not self.applied:
            raise RuntimeError("EMA parameters are not applied.")
        for name, parameter in self.referenced.items():
            parameter.copy_(self.backup[name].to(device=parameter.device, dtype=parameter.dtype))
        self.backup.clear()

    def state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in self.shadow.items()
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> None:
        source_keys = set(state_dict)
        target_keys = set(self.shadow)
        if strict:
            missing_keys = sorted(target_keys - source_keys)
            unexpected_keys = sorted(source_keys - target_keys)
            if missing_keys or unexpected_keys:
                messages = []
                if missing_keys:
                    messages.append("Missing EMA keys:\n" + "\n".join(f"  {key}" for key in missing_keys))
                if unexpected_keys:
                    messages.append("Unexpected EMA keys:\n" + "\n".join(f"  {key}" for key in unexpected_keys))
                raise KeyError("\n".join(messages))
        for name in target_keys & source_keys:
            source = state_dict[name]
            target = self.shadow[name]
            if source.shape != target.shape:
                raise RuntimeError(
                    f"EMA shape mismatch for '{name}': expected {tuple(target.shape)}, "
                    f"got {tuple(source.shape)}."
                )
            self.shadow[name] = source.detach().clone().to(
                device=target.device, dtype=target.dtype
            )
