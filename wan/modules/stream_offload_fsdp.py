"""stream_offload_fsdp.py
=====================================================
Lightweight helper that re‑uses the original OffloadManager to
provide **stream‑offload** capability for **FSDP‑wrapped** models.

‑ Detects whether the model has already been wrapped by FSDP.
‑ Collects each FSDP wrapper corresponding to a Transformer block.
‑ Instantiates OffloadManager with those wrappers so that *each rank*
can asynchronously move its shard CPU⇄GPU just in time for the
All‑Gather that FSDP will perform.

Usage (inside wan/vace.py, after the model has been wrapped by FSDP)::

    if offload_model and dist.is_initialized():
        from wan.modules.stream_offload_fsdp import enable_fsdp_stream_offload
        self._offloader = enable_fsdp_stream_offload(
            self.model,
            keep_n=1,
            device=self.device,
        )

Keeping the helper isolated avoids touching the original
``stream_offload.py`` logic while enabling easy experimentation.
"""

from __future__ import annotations

import types
from typing import List, Sequence, Dict, Any

import torch
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
except Exception:  # pragma: no cover – older torch
    FSDP = None  # type: ignore

from .stream_offload import OffloadManager  # original single‑GPU manager

__all__ = ["enable_fsdp_stream_offload"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_fsdp_instance(m: torch.nn.Module) -> bool:
    """Return True if *m* is an actual FSDP wrapper instance."""
    return FSDP is not None and isinstance(m, FSDP)


def _collect_fsdp_blocks(root: torch.nn.Module) -> List[torch.nn.Module]:
    """Locate the list of FSDP‑wrapped transformer blocks.

    Wan2.1 keeps blocks either at ``model.blocks`` (no FSDP) or at
    ``model.module.blocks`` (after wrapping the whole model in FSDP once).
    Each element **inside** that list is itself an FSDP wrapper around the
    *real* block.  We just return those wrappers so that OffloadManager
    can attach forward hooks directly to them.
    """
    if hasattr(root, "module") and hasattr(root.module, "blocks"):
        return list(root.module.blocks)  # type: ignore[arg‑type]
    if hasattr(root, "blocks"):
        return list(root.blocks)  # type: ignore[arg‑type]
    raise AttributeError("Could not find 'blocks' attribute on model – layout may have changed.")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def enable_fsdp_stream_offload(
    model: torch.nn.Module,
    *,
    keep_n: int = 1,
    device: torch.device | None = None,
) -> OffloadManager:
    """Attach **stream‑offload** hooks to an already FSDP‑wrapped *model*.

    Parameters
    ----------
    model
        The model root that has **already** been wrapped by FSDP.
    keep_n
        Sliding‑window size in GPU layers; equivalent to OffloadManager.keep_n.
    device
        GPU device where the model shards live (defaults to current).

    Returns
    -------
    OffloadManager
        The active manager instance.  *Keep a reference alive* to avoid
        hooks getting garbage‑collected.
    """

    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized; FSDP stream offload only makes sense when torch.distributed is active.")

    blocks: Sequence[torch.nn.Module] = _collect_fsdp_blocks(model)

    module_groups: Dict[str, Sequence[torch.nn.Module]] = {"blocks": blocks}

    offloader = OffloadManager(
        root=model,
        module_groups=module_groups,
        keep_n=keep_n,
        device=device,
    )
    offloader.enable()
    return offloader
