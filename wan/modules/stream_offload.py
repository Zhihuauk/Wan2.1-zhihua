import torch
import weakref
from typing import Dict, Sequence, Optional

def _is_fsdp(module):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(module, FSDP)
    except ImportError:
        return False

def _is_ddp(module):
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return isinstance(module, DDP)
    except ImportError:
        return False

class OffloadManager:
    """
    通用流式权重管理器，支持单卡和分布式（FSDP/DDP）推理。
    """

    def __init__(
        self,
        root: torch.nn.Module,
        module_groups: Dict[str, Sequence[torch.nn.Module]],
        keep_n: Dict[str, int] | int,
        device: Optional[torch.device] = None,
        distributed: bool = False,
    ):
        self.root = weakref.proxy(root)
        self.device = device or next(root.parameters()).device
        self.keep_n = (
            {k: keep_n for k in module_groups} if isinstance(keep_n, int) else keep_n
        )
        self.groups = module_groups
        self.distributed = distributed

        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()
        self.enabled = False
        self.handles = []

        # 递归注册 index/depth
        for name, grp in self.groups.items():
            for i, m in enumerate(grp):
                setattr(m, "_stream_offload_index", i)
                setattr(m, "_stream_offload_depth", len(grp))

    def enable(self):
        if self.enabled:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for streaming-offload.")

        # 0. 保证前 keep_n 层在 self.device
        for tag, grp in self.groups.items():
            stay_on_gpu = min(self.keep_n[tag], len(grp))
            for i in range(stay_on_gpu):
                self._move_module_to_device(grp[i], self.device)

        # 1. keep_n 之后的层做 offload
        for tag, grp in self.groups.items():
            if len(grp) > self.keep_n[tag]:
                self._setup_parameter_offload(grp, self.keep_n[tag])

        # 2. 注册 hooks
        for tag, grp in self.groups.items():
            for m in grp:
                pre_h = m.register_forward_pre_hook(
                    self._prefetch_hook_factory(tag), with_kwargs=False
                )
                self.handles.append(pre_h)
                if getattr(m, "_stream_offload_index") >= self.keep_n[tag]:
                    post_h = m.register_forward_hook(
                        self._release_hook_factory(tag), with_kwargs=False
                    )
                    self.handles.append(post_h)

        self.enabled = True
        print(f"[stream_offload] enabled (keep_n={self.keep_n}, device={self.device}, distributed={self.distributed})")

    def disable(self):
        if not self.enabled:
            return
        self._restore_all()
        self._remove_hooks()
        self.enabled = False
        print("[stream_offload] disabled – all weights back on GPU")

    # --- internal ---

    def _move_module_to_device(self, module, device):
        # FSDP/DDP wrapper下递归到实际层
        if _is_fsdp(module) or _is_ddp(module):
            self._move_module_to_device(module.module, device)
        else:
            module.to(device, non_blocking=False)

    @staticmethod
    def _iter_tensors(m: torch.nn.Module):
        yield from m.parameters(recurse=True)
        yield from m.buffers(recurse=True)

    def _prefetch_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        grp = self.groups[tag]

        def hook(module, _inputs):
            idx = getattr(module, "_stream_offload_index")
            if idx + keep_n >= len(grp):
                return
            nxt = grp[idx + keep_n]
            evt = torch.cuda.Event(); evt.record()
            with torch.cuda.stream(self.h2d_stream):
                self.h2d_stream.wait_event(evt)
                for p in self._iter_tensors(nxt):
                    self._ensure_gpu_tensor(p)
            # torch.cuda.current_stream().wait_stream(self.h2d_stream)
        return hook

    def _release_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        def hook(module, _inp, _out):
            idx = getattr(module, "_stream_offload_index")
            if idx < keep_n:
                return
            for p in self._iter_tensors(module):
                self._release_tensor(p)
        return hook

    def _setup_parameter_offload(self, grp, keep_n):
        for idx in range(keep_n, len(grp)):
            blk = grp[idx]
            with torch.cuda.stream(self.d2h_stream):
                for p in self._iter_tensors(blk):
                    if not hasattr(p, "_stream_offload_p_cpu"):
                        # 只保存本 rank 的 shard
                        p._stream_offload_p_cpu = torch.empty_like(p.data, pin_memory=True, device="cpu")
                        p._stream_offload_orig_shape = tuple(p.shape)
                    p._stream_offload_p_cpu.copy_(p.data, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.d2h_stream)
        for idx in range(keep_n, len(grp)):
            blk = grp[idx]
            for p in self._iter_tensors(blk):
                self._release_tensor(p)

    def _ensure_gpu_tensor(self, p: torch.Tensor):
        dev = self.device
        need_alloc = (
            p.data.device != dev or
            p.data.untyped_storage().size() == 0 or
            p.data.untyped_storage().size() < getattr(p, "_stream_offload_storage_size", p.numel())
        )
        if need_alloc:
            target_shape = getattr(p, "_stream_offload_orig_shape", tuple(
                p._stream_offload_p_cpu.shape if hasattr(p, "_stream_offload_p_cpu") else p.shape)
            )
            p.data = torch.empty(target_shape, dtype=p.dtype, device=dev)
        if hasattr(p, "_stream_offload_p_cpu"):
            p.data.copy_(p._stream_offload_p_cpu, non_blocking=True)
        else:
            if need_alloc or p.data.device != dev:
                p.data = p.data.to(dev, non_blocking=True)
        if getattr(p, "_stream_offload_released", False):
            p._stream_offload_released = False

    @staticmethod
    def _release_tensor(p: torch.Tensor):
        try:
            p.data.untyped_storage().resize_(0)
            p.data.resize_(0)
            p._stream_offload_released = True
        except Exception:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
            p._stream_offload_released = True

    def _restore_all(self):
        for grp in self.groups.values():
            for m in grp:
                for p in self._iter_tensors(m):
                    if hasattr(p, "_stream_offload_p_cpu"):
                        if p.data.untyped_storage().size() == 0:
                            target = torch.empty(p._stream_offload_orig_shape, dtype=p.dtype, device=self.device)
                            target.copy_(p._stream_offload_p_cpu, non_blocking=False)
                            p.data = target
                        elif p.data.device != self.device:
                            p.data = p.data.to(self.device, non_blocking=False)

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
