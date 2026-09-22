"""Helpers for the note/syllable groups used by duration prediction.

A group is the unit whose total duration is fixed by the score: the phones of a
note or syllable share one frame budget, and the predictor only decides how that
budget is split. Training marks groups with ``ph2word`` (index 0 means padding),
inference derives them from ``word_div`` with the length regulator.

All helpers are vectorized and use only export-safe tensor ops, so the same code
path serves training and the exported graph.
"""

import torch
from torch import Tensor

__all__ = [
    "group_mask",
    "group_sizes",
    "group_position_ids",
    "group_distribution",
    "group_log_prob",
    "allocate_counts",
]

# Large negative logit used instead of -inf so that empty rows stay finite.
_MASKED_LOGIT = -1e4
_EPS = 1e-8


def group_mask(group_ids: Tensor, num_groups=None, x_masks: Tensor = None) -> Tensor:
    """One-hot membership mask.

    :param group_ids: [B, T] group index per item, 1-based, 0 for padding
    :param num_groups: number of groups; when omitted it is taken from the
        largest index in ``group_ids``, which keeps the mask compatible with a
        dynamic number of groups
    :param x_masks: [B, T] bool mask, True for padded items
    :return: [B, T, W] bool mask
    """
    if num_groups is None:
        num_groups = group_ids.max()
    index = torch.arange(1, num_groups + 1, device=group_ids.device)
    mask = group_ids[:, :, None] == index[None, None, :]
    if x_masks is not None:
        mask = mask & ~x_masks.bool()[:, :, None]
    return mask


def group_sizes(mask: Tensor) -> Tensor:
    """:param mask: [B, T, W] membership mask; :return: [B, W] items per group"""
    return mask.sum(dim=1)


def group_position_ids(mask: Tensor, max_position: int = None):
    """Positions of every item inside its group, counted forward and backward.

    :param mask: [B, T, W] membership mask
    :param max_position: optional clamp, so long groups saturate instead of
        growing the embedding table
    :return: (forward, reverse) [B, T] long tensors; padded items map to 0
    """
    numeric = mask.to(torch.int64)  # CumSum has no boolean kernel
    running = numeric.cumsum(dim=1) * numeric  # [B, T, W]
    forward = running.sum(dim=2) - 1  # [B, T]
    size = (group_sizes(mask)[:, None, :] * numeric).sum(dim=2)  # [B, T]
    reverse = size - 1 - forward
    forward = forward.clamp_min(0)
    reverse = reverse.clamp_min(0)
    if max_position is not None:
        forward = forward.clamp_max(max_position)
        reverse = reverse.clamp_max(max_position)
    return forward, reverse


def group_distribution(duration: Tensor, ph2word: Tensor, mask: Tensor = None) -> Tensor:
    """Normalize durations into a distribution over the phones of each group.

    :param duration: [B, T] durations or scores
    :param ph2word: [B, T] group index per item, 0 for padding
    :param mask: optional [B, T] bool mask of items to include; by default every
        item with a positive group index is included
    :return: [B, T] probability of each item inside its group, zero elsewhere
    """
    if mask is None:
        mask = ph2word > 0
    shape = duration.shape[0], ph2word.max() + 1
    total = duration.new_zeros(*shape).scatter_add(
        1, ph2word, duration * mask
    )[:, 1:]  # [B, T] => [B, T_w]
    per_group = total.gather(1, ph2word.clamp(min=1) - 1)  # group total per item
    return duration * mask / per_group.clamp_min(1e-8)


def group_log_prob(logits: Tensor, mask: Tensor) -> Tensor:
    """Log softmax of ``logits`` normalized within each group.

    :param logits: [B, T] unnormalized scores
    :param mask: [B, T, W] membership mask, padded items must be unset
    :return: [B, T] log probability of each item inside its group
    """
    expanded = logits[:, :, None].masked_fill(~mask, _MASKED_LOGIT)  # [B, T, W]
    group_max = expanded.max(dim=1).values  # [B, W]
    shifted = (expanded - group_max[:, None, :]).exp() * mask
    log_norm = group_max + shifted.sum(dim=1).clamp_min(_EPS).log()  # [B, W]
    return ((expanded - log_norm[:, None, :]) * mask).sum(dim=2)  # [B, T]


def allocate_counts(prob: Tensor, budget: Tensor, x_masks: Tensor = None) -> Tensor:
    """Turn within-group probabilities and per-item budgets into frame counts.

    The cumulative probability is scaled by the budget, rounded with ties away
    from zero and then differenced. Because the probabilities of a group sum to
    one, the boundary at the end of a group equals that group's budget exactly,
    so the counts of every group sum to its budget without any repair step.

    :param prob: [B, T] within-group probabilities
    :param budget: [B, T] budget in frames of the group each item belongs to; it
        must be constant inside a group, so build it by gathering the per-group
        budget with the group ids
    :param x_masks: [B, T] bool mask, True for padded items
    :return: [B, T] frame counts
    """
    cumulative = (prob.clamp_min(0.) * budget).cumsum(dim=1)
    boundary = (cumulative + 0.5).floor()
    previous = torch.cat([boundary.new_zeros(boundary.shape[0], 1), boundary[:, :-1]], dim=1)
    counts = boundary - previous
    if x_masks is not None:
        counts = counts.masked_fill(x_masks.bool(), 0.)
    return counts
