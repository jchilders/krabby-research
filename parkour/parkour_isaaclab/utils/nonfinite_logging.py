# SPDX-License-Identifier: BSD-3-Clause
"""Rate-limited warnings when tensors contain NaN/Inf before sanitization."""

from __future__ import annotations

import logging

import torch

_LOG = logging.getLogger(__name__)

_MAX_MESSAGES_PER_TAG = 3
_counts: dict[str, int] = {}


def warn_if_nonfinite(tag: str, t: torch.Tensor) -> None:
    """Log at most _MAX_MESSAGES_PER_TAG warnings per tag when ``t`` has non-finite values."""
    if t.numel() == 0 or torch.isfinite(t).all():
        return
    n = _counts.get(tag, 0)
    if n >= _MAX_MESSAGES_PER_TAG:
        return
    _counts[tag] = n + 1
    bad = int((~torch.isfinite(t)).sum().item())
    _LOG.warning(
        "Non-finite values before sanitize [%s]: shape=%s nonfinite_elements=%d (message %d/%d for this tag)",
        tag,
        tuple(t.shape),
        bad,
        n + 1,
        _MAX_MESSAGES_PER_TAG,
    )
