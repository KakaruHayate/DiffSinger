from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _repeat_batch(value, repeats: int):
    if isinstance(value, tuple):
        return tuple(torch.cat([item] * repeats, dim=0) for item in value)
    if isinstance(value, list):
        return [torch.cat([item] * repeats, dim=0) for item in value]
    return torch.cat([value] * repeats, dim=0)


def run_reflow_xm(
        predictor: nn.Module,
        condition: Tensor,
        target: Tensor | Sequence[Tensor],
        loss_fn: nn.Module,
        non_padding: Tensor | None,
        best_of_k: int,
        chunk_size: int,
):
    """Run low-memory Forward XM for a Rectified Flow predictor.

    Candidate noises are evaluated without gradients in bounded chunks. All
    candidates for an original sample share one timestep. The winning noise is
    replayed once with gradients so upstream condition encoders remain trainable.
    """
    if best_of_k < 2:
        raise ValueError('run_reflow_xm requires best_of_k >= 2.')
    if chunk_size < 1:
        raise ValueError('XM chunk_size must be at least 1.')

    spec, timestep, _ = predictor.prepare_training_inputs(target)
    batch_size = spec.shape[0]
    best_losses = torch.full((batch_size,), float('inf'), device=spec.device)
    best_noise = torch.empty_like(spec)

    dropout_modules = [
        module for module in predictor.modules()
        if isinstance(module, nn.Dropout) and module.p > 0.
    ]
    if dropout_modules:
        raise RuntimeError(
            'Explorative Modeling save-memory replay requires predictor backbone dropout_rate=0.'
        )

    with torch.no_grad():
        for chunk_start in range(0, best_of_k, chunk_size):
            chunk_candidates = min(chunk_size, best_of_k - chunk_start)
            noise = torch.randn(
                (chunk_candidates, *spec.shape),
                device=spec.device,
                dtype=spec.dtype,
            )
            chunk_condition = torch.cat([condition.detach()] * chunk_candidates, dim=0)
            chunk_target = _repeat_batch(target, chunk_candidates)
            chunk_timestep = torch.cat([timestep] * chunk_candidates, dim=0)
            chunk_non_padding = None if non_padding is None else torch.cat(
                [non_padding] * chunk_candidates, dim=0
            )

            v_pred, v_gt, _ = predictor.training_forward(
                chunk_condition,
                chunk_target,
                t=chunk_timestep,
                noise=noise.flatten(0, 1),
            )
            chunk_losses = loss_fn(
                v_pred,
                v_gt,
                t=chunk_timestep,
                non_padding=chunk_non_padding,
                reduction='none',
            ).reshape(chunk_candidates, batch_size)
            chunk_best_losses, chunk_best_indices = chunk_losses.min(dim=0)
            batch_indices = torch.arange(batch_size, device=spec.device)
            chunk_best_noise = noise[chunk_best_indices, batch_indices]
            replace = chunk_best_losses < best_losses
            best_losses[replace] = chunk_best_losses[replace]
            best_noise[replace] = chunk_best_noise[replace]

    if hasattr(torch, 'clear_autocast_cache'):
        torch.clear_autocast_cache()
    return predictor.training_forward(
        condition,
        target,
        t=timestep,
        noise=best_noise,
    )
