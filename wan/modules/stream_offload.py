import logging
import weakref
from typing import Dict, Sequence, List, Union, Optional

import torch
import torch.nn as nn

# ==================== helpers ====================

def _is_fsdp(module: nn.Module) -> bool:
    """Return True if *module* is an FSDP wrapper."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: autoimport
        return isinstance(module, FSDP)
    except ImportError:
        return False


def _is_ddp(module: nn.Module) -> bool:
    """Return True if *module* is a DDP wrapper."""
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: autoimport
        return isinstance(module, DDP)
    except ImportError:
        return False


def _iter_module_tensors(module: nn.Module):
    """Yield all real memory–owning tensors (params + buffers) inside *module*.

    For FSDP with ``use_orig_params=True`` we can access parameters directly.
    For DDP we unwrap one level via ``module.module``.
    """
    if _is_fsdp(module):
        yield from module.parameters(recurse=True)
        yield from module.buffers(recurse=True)
    elif _is_ddp(module):
        yield from module.module.parameters(recurse=True)
        yield from module.module.buffers(recurse=True)
    else:
        yield from module.parameters(recurse=True)
        yield from module.buffers(recurse=True)


# ==================== OffloadManager ====================

class OffloadManager:
    """Streaming weight off‑load manager.

    The API is *compatible with the original single‑card version* while
    internally adopting some of the clearer abstractions from the multi‑card
    implementation.

    Parameters
    ----------
    root : nn.Module
        The root model (only kept as weakref to break reference cycles).
    module_groups : Dict[str, Sequence[nn.Module]] | None
        Mapping name → list of sequential sub‑modules (e.g. transformer blocks).
        Can be omitted and later registered via :py:meth:`register_modules`.
    keep_n : int | Dict[str, int]
        How many leading blocks in each group should *stay* on GPU.
        Int applies to all groups; dict allows per‑group config.
    device : torch.device | None
        Destination CUDA device. *None* → device of the first parameter.
    distributed : bool
        Flag to indicate FSDP/DDP environment (affects memory accounting but
        does not change public API).
    """

    def __init__(
        self,
        root: nn.Module,
        module_groups: Optional[Dict[str, Sequence[nn.Module]]] = None,
        keep_n: Union[int, Dict[str, int]] = 2,
        device: Optional[torch.device] = None,
        distributed: bool = False,
    ) -> None:
        # ── basic attrs ────────────────────────────────────────────────────
        self.root = weakref.proxy(root)
        self.device = device or next(root.parameters()).device
        self.distributed = distributed
        self.enabled = False

        # ── module registry ───────────────────────────────────────────────
        self.groups: Dict[str, List[nn.Module]] = {}
        if module_groups:
            for name, modules in module_groups.items():
                self.register_modules(name, modules, keep_n if isinstance(keep_n, int) else keep_n.get(name, 2))

        # keep_n becomes dict[str,int] after registration; temp store original
        self._keep_n_default = keep_n

        # ── streams & hook handles ────────────────────────────────────────
        self.h2d_stream: Optional[torch.cuda.Stream] = None
        self.d2h_stream: Optional[torch.cuda.Stream] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        # ── logger ────────────────────────────────────────────────────────
        self.logger = logging.getLogger(__name__)

    # ====================================================================
    # public API
    # ====================================================================

    def register_modules(
        self,
        name: str,
        modules: Sequence[nn.Module],
        resident_count: int | None = None,
    ) -> None:
        """Register a *sequential* block list that will participate in off‑load."""
        if self.enabled:
            raise RuntimeError("Cannot register new modules after enable(). Call disable() first.")
        if name in self.groups:
            raise ValueError(f"Duplicate module group name: {name!r}")
        self.groups[name] = list(modules)
        # init keep_n mapping
        if not hasattr(self, "keep_n"):
            # first registration → build keep_n dict later
            pass
        if isinstance(self._keep_n_default, int):
            k = self._keep_n_default if resident_count is None else resident_count
        else:
            k = self._keep_n_default.get(name, resident_count or 2)
        if not hasattr(self, "keep_n"):
            self.keep_n: Dict[str, int] = {}
        self.keep_n[name] = k
        # annotate helpers
        for i, m in enumerate(modules):
            setattr(m, "_stream_offload_index", i)
            setattr(m, "_stream_offload_depth", len(modules))
            setattr(m, "_stream_offload_group", name)
        self.logger.info(
            "[stream_offload] registered group '%s': %d modules, keep_n=%d",
            name, len(modules), k,
        )

    # --------------------------------------------------------------------
    def enable(self) -> None:
        """Activate streaming off‑load."""
        if self.enabled:
            self.logger.warning("stream_offload already enabled – skipping …")
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for streaming‑offload.")
        if not self.groups:
            raise RuntimeError("No module_groups registered.")

        # create streams only now (lazy) to use current device context
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

        # step‑0: ensure first *keep_n* blocks on GPU
        for tag, grp in self.groups.items():
            for idx in range(min(self.keep_n[tag], len(grp))):
                self._move_module_to_device(grp[idx], self.device)

        # step‑1: warm‑up off‑load non‑resident blocks
        for tag, grp in self.groups.items():
            if len(grp) > self.keep_n[tag]:
                self._initial_offload(grp, self.keep_n[tag])

        # step‑2: register pre/post hooks
        for tag, grp in self.groups.items():
            for m in grp:
                pre = m.register_forward_pre_hook(self._prefetch_hook_factory(tag))
                self._handles.append(pre)
                if getattr(m, "_stream_offload_index") >= self.keep_n[tag]:
                    pst = m.register_forward_hook(self._release_hook_factory(tag))
                    self._handles.append(pst)

        self.enabled = True
        self.logger.info("[stream_offload] enabled on %s (groups=%d).", self.device, len(self.groups))

    # --------------------------------------------------------------------
    def disable(self, restore: bool = True) -> None:
        """Disable and (optionally) bring all parameters back to GPU."""
        if not self.enabled:
            return
        # remove hooks
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if restore:
            self._restore_all()
        self.enabled = False
        self.logger.info("[stream_offload] disabled – all hooks cleared%s.", " and weights restored" if restore else "")

    # alias for context manager compat with old API
    __enter__ = lambda self: (self.enable() or self)
    __exit__ = lambda self, *exc: self.disable()

    # ====================================================================
    # internals – off‑load mechanics
    # ====================================================================

    def _initial_offload(self, grp: Sequence[nn.Module], keep_n: int):
        """Copy parameters of blocks >= *keep_n* to CPU‑pinned memory and free GPU."""
        assert self.d2h_stream is not None  # created in enable()
        for blk in grp[keep_n:]:
            with torch.cuda.stream(self.d2h_stream):
                for p in _iter_module_tensors(blk):
                    if not hasattr(p, "_so_p_cpu"):
                        # attr prefix _so_*  (stream offload) to avoid clashes
                        p._so_p_cpu = torch.empty_like(p.data, pin_memory=True, device="cpu")
                        p._so_orig_shape = tuple(p.shape)
                        p._so_storage_size = p.data.untyped_storage().size()
                        p._so_is_slice = p.data.untyped_storage().size() != p.data.numel()
                    # copy data
                    if p._so_is_slice:
                        p._so_p_cpu.copy_(p.data, non_blocking=True)
                    else:
                        p._so_p_cpu.untyped_storage().copy_(p.data.untyped_storage(), non_blocking=True)
            # synchronize copy before releasing storage
            torch.cuda.current_stream().wait_stream(self.d2h_stream)

        # free GPU memory
        for blk in grp[keep_n:]:
            for p in _iter_module_tensors(blk):
                self._release_tensor(p)

    # --------------------------------------------------------------------
    def _prefetch_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        grp = self.groups[tag]

        def _hook(m: nn.Module, _inputs):
            idx = getattr(m, "_stream_offload_index")
            nxt_idx = idx + keep_n
            if nxt_idx >= len(grp):
                return  # nothing to prefetch
            next_blk = grp[nxt_idx]
            event = torch.cuda.Event(); event.record()
            assert self.h2d_stream is not None
            with torch.cuda.stream(self.h2d_stream):
                self.h2d_stream.wait_event(event)
                for p in _iter_module_tensors(next_blk):
                    self._ensure_gpu_tensor(p)
        return _hook

    # --------------------------------------------------------------------
    def _release_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        grp = self.groups[tag]

        def _hook(m: nn.Module, _inp, _out):
            idx = getattr(m, "_stream_offload_index")
            if idx < keep_n:
                return  # resident block
            for p in _iter_module_tensors(m):
                self._release_tensor(p)
        return _hook

    # --------------------------------------------------------------------
    def _ensure_gpu_tensor(self, p: torch.Tensor):
        dev = self.device
        need_alloc = (
            p.data.device != dev or
            p.data.untyped_storage().size() == 0 or
            p.data.untyped_storage().size() < getattr(p, "_so_storage_size", p.numel())
        )
        if need_alloc:
            target_shape = getattr(p, "_so_orig_shape", tuple(p.shape))
            p.data = torch.empty(target_shape, dtype=p.dtype, device=dev)
            if hasattr(p, "_so_storage_size"):
                try:
                    if p._so_storage_size != p.data.untyped_storage().size():
                        p.data.untyped_storage().resize_(p._so_storage_size)
                except Exception:
                    pass  # fall back silently
        # copy back from CPU cache if available
        if hasattr(p, "_so_p_cpu"):
            try:
                p.data.copy_(p._so_p_cpu, non_blocking=True)
            except Exception:
                if not getattr(p, "_so_is_slice", False):
                    try:
                        p.data.untyped_storage().copy_(p._so_p_cpu.untyped_storage(), non_blocking=True)
                    except Exception:
                        pass
        if getattr(p, "_so_released", False):
            p._so_released = False

    # --------------------------------------------------------------------
    @staticmethod
    def _release_tensor(p: torch.Tensor):
        try:
            p.data.untyped_storage().resize_(0)
            p.data.resize_(0)
            p._so_released = True
        except Exception:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
            p._so_released = True

    # --------------------------------------------------------------------
    def _restore_all(self):
        for grp in self.groups.values():
            for blk in grp:
                for p in _iter_module_tensors(blk):
                    if hasattr(p, "_so_p_cpu"):
                        if p.data.untyped_storage().size() == 0:
                            target = torch.empty(p._so_orig_shape, dtype=p.dtype, device=self.device)
                            try:
                                if p._so_storage_size != target.untyped_storage().size():
                                    target.untyped_storage().resize_(p._so_storage_size)
                            except Exception:
                                pass
                            try:
                                target.copy_(p._so_p_cpu, non_blocking=False)
                            except Exception:
                                if not p._so_is_slice:
                                    try:
                                        target.untyped_storage().copy_(p._so_p_cpu.untyped_storage(), non_blocking=False)
                                    except Exception:
                                        pass
                            p.data = target
                        elif p.data.device != self.device:
                            p.data = p.data.to(self.device, non_blocking=False)

    # ====================================================================
    # utils
    # ====================================================================

    @staticmethod
    def _move_module_to_device(module: nn.Module, device: torch.device):
        if _is_fsdp(module):
            module.to(device, non_blocking=False)
        elif _is_ddp(module):
            module.module.to(device, non_blocking=False)
        else:
            module.to(device, non_blocking=False)

    # --------------------------------------------------------------------
    def get_memory_stats(self):
        if not torch.cuda.is_available():
            return {"gpu_memory": "N/A"}
        alloc = torch.cuda.memory_allocated(self.device) / 1024 ** 3
        reserv = torch.cuda.memory_reserved(self.device) / 1024 ** 3
        total_blks = sum(len(g) for g in self.groups.values())
        resident_blks = sum(self.keep_n.values())
        return {
            "gpu_memory_allocated": f"{alloc:.2f} GB",
            "gpu_memory_reserved": f"{reserv:.2f} GB",
            "offload_enabled": self.enabled,
            "total_blocks": total_blks,
            "resident_blocks": resident_blks,
            "offloaded_blocks": total_blks - resident_blks,
            "keep_n": dict(self.keep_n),
            "device": str(self.device),
        }

    # --------------------------------------------------------------------
    def set_keep_n(self, new_keep_n: Union[int, Dict[str, int]]):
        if isinstance(new_keep_n, int):
            self.keep_n = {k: new_keep_n for k in self.groups}
        else:
            self.keep_n.update(new_keep_n)
        if self.enabled:
            self.logger.warning("keep_n updated but offload already enabled. Call disable() & enable() to apply.")

    # --------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return self.enabled
