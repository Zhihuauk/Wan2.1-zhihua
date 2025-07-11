# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from .model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d


class VaceWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VaceWanModel(WanModel):

    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='vace',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim,
                         freq_dim, text_dim, out_dim, num_heads, num_layers,
                         window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)
                           ] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {
            i: n for n, i in enumerate(self.vace_layers)
        }

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=self.vace_layers_mapping[i]
                if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=i) for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size)

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        hints = []
        for block in self.vace_blocks:
            c, c_skip = block(c, **new_kwargs)
            hints.append(c_skip)
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

#=======================流式加载版本 v0.5，模型和流加载解耦合============
# stream_offload.py
# from __future__ import annotations
import torch, weakref
from typing import Dict, List, Sequence, Tuple

class OffloadManager:
    """
    Generic streaming-weight offloader.
    Example
    -------
    model = VaceWanModel(...)
    offloader = OffloadManager(
        model,
        module_groups={
            "blocks": model.blocks,
            "vace_blocks": model.vace_blocks,
        },
        keep_n={"blocks": 10, "vace_blocks": 10},
    )
    offloader.enable()      # 推理
    ...
    offloader.disable()     # 结束
    """

    def __init__(
        self,
        root: torch.nn.Module,
        module_groups: Dict[str, Sequence[torch.nn.Module]],
        keep_n: Dict[str, int] | int,
        device: torch.device | None = None,
    ):
        self.root        = weakref.proxy(root)   # 不与模型相互引用
        self.device      = device or next(root.parameters()).device
        self.keep_n      = (
            {k: keep_n for k in module_groups} if isinstance(keep_n, int) else keep_n
        )
        self.groups      = module_groups

        self.h2d_stream  = torch.cuda.Stream()
        self.d2h_stream  = torch.cuda.Stream()
        self.enabled     = False
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        # 保证每层都有 index/depth 属性（如果模型没设）
        for name, grp in self.groups.items():
            for i, m in enumerate(grp):
                if not hasattr(m, "index"):
                    m.index = i
                if not hasattr(m, "depth"):
                    m.depth = len(grp)

    def enable(self):
        """
        打开流式权重加载 / 推理 offload。
        逻辑顺序：
            0) 先把“常驻层”(前 keep_n) 全量搬到目标 GPU
            1) 针对其余层执行 CPU-offload（复制到 pinned-CPU 并释放 GPU）
            2) 注册前向 pre / post hooks
        """
        if self.enabled:                       # 已开启则直接返回
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for streaming-offload.")

        # --------------------------------------------------
        # 0️⃣  确保前 keep_n 层全部在 self.device
        # --------------------------------------------------
        for tag, grp in self.groups.items():
            stay_on_gpu = min(self.keep_n[tag], len(grp))
            for i in range(stay_on_gpu):
                grp[i].to(self.device, non_blocking=False)

        # --------------------------------------------------
        # 1️⃣  对 keep_n 之后的层做 offload
        # --------------------------------------------------
        for tag, grp in self.groups.items():
            if len(grp) > self.keep_n[tag]:
                self._setup_parameter_offload(grp, self.keep_n[tag])

        # --------------------------------------------------
        # 2️⃣  注册 hooks：预取 + 释放
        # --------------------------------------------------
        for tag, grp in self.groups.items():
            for m in grp:
                # forward 之前预取下一层
                pre_h = m.register_forward_pre_hook(
                    self._prefetch_hook_factory(tag), with_kwargs=False
                )
                self.handles.append(pre_h)

                # forward 结束后释放自己（仅超出 keep_n 的层）
                if m.index >= self.keep_n[tag]:
                    post_h = m.register_forward_hook(
                        self._release_hook_factory(tag), with_kwargs=False
                    )
                    self.handles.append(post_h)

        # --------------------------------------------------
        # ✅  启动后做一次快速校验
        # --------------------------------------------------
        for tag, grp in self.groups.items():
            for i in range(min(self.keep_n[tag], len(grp))):
                p_dev = next(grp[i].parameters()).device
                assert p_dev == self.device, (
                    f"{tag}[{i}] still on {p_dev}, expect {self.device}"
                )

        self.enabled = True
        print(f"[offload] enabled on {list(self.groups)} "
            f"(keep_n={self.keep_n}, device={self.device})")


    def disable(self):
        if not self.enabled:
            return
        self._restore_all()
        self._remove_hooks()
        self.enabled = False
        print("[offload] disabled – all weights back on GPU")

    # ---------- internal ----------
    # a. 迭代张量
    @staticmethod
    def _iter_tensors(m: torch.nn.Module):
        yield from m.parameters(recurse=True)
        yield from m.buffers(recurse=True)

    # b. 预取 hook
    def _prefetch_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        grp    = self.groups[tag]

        def hook(module, _inputs):
            if module.index + keep_n >= len(grp):
                return                      # 最后几层无需预取

            nxt = grp[module.index + keep_n]
            evt = torch.cuda.Event(); evt.record()

            with torch.cuda.stream(self.h2d_stream):
                self.h2d_stream.wait_event(evt)
                for p in self._iter_tensors(nxt):
                    self._ensure_gpu_tensor(p)
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
        return hook

    # c. 释放 hook
    def _release_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]

        def hook(module, _inp, _out):
            if module.index < keep_n:
                return
            for p in self._iter_tensors(module):
                self._release_tensor(p)
        return hook

    # d. CPU offload upfront
    def _setup_parameter_offload(self, grp, keep_n):
        for idx in range(keep_n, len(grp)):
            blk = grp[idx]
            with torch.cuda.stream(self.d2h_stream):
                for p in self._iter_tensors(blk):
                    if not hasattr(p, "p_cpu"):
                        p.p_cpu = torch.empty_like(p.data, pin_memory=True, device="cpu")
                    p.p_cpu.copy_(p.data, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.d2h_stream)
        for idx in range(keep_n, len(grp)):
            blk = grp[idx]
            for p in self._iter_tensors(blk):
                self._release_tensor(p)

    # e. helpers
    def _ensure_gpu_tensor(self, p: torch.Tensor):
        """
        确保张量 p 位于 self.device 并具有正确 shape / storage。
        若 p 被释放过（storage==0）或大小不足，重新分配；随后把
        备份的 p_cpu 数据复制到 GPU 张量。
        """
        dev = self.device

        # ① 判断是否需要重新分配
        need_alloc = (
            p.data.device != dev or                           # 不在目标 GPU
            p.data.untyped_storage().size() == 0 or           # storage 已清零
            p.data.untyped_storage().size() <
            getattr(p, "storage_size", p.numel())             # storage 太小
        )

        if need_alloc:
            # 取原始形状（在 _setup_parameter_offload() 中记录）
            target_shape = getattr(p, "orig_shape", tuple(
                p.p_cpu.shape if hasattr(p, "p_cpu") else p.shape)
            )
            # 重新分配 GPU 张量
            p.data = torch.empty(target_shape, dtype=p.dtype, device=dev)

        # ② 把备份数据同步到 GPU
        if hasattr(p, "p_cpu"):
            if getattr(p, "is_slice_tensor", False):
                p.data.copy_(p.p_cpu, non_blocking=True)
            else:
                p.data.untyped_storage().copy_(
                    p.p_cpu.untyped_storage(), non_blocking=True
                )
        else:
            # 首次出现 / buffer：只需保证在 GPU
            if need_alloc or p.data.device != dev:
                p.data = p.data.to(dev, non_blocking=True)

        # ③ 清除 “已释放” 标记
        if getattr(p, "_released", False):
            p._released = False

    @staticmethod
    def _release_tensor(p: torch.Tensor):
        try:    # view 友好
            p.data.untyped_storage().resize_(0)
            p.data.resize_(0)
        except RuntimeError:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    def _restore_all(self):
        for grp in self.groups.values():
            for m in grp:
                for p in self._iter_tensors(m):
                    if p.data.untyped_storage().size() == 0:
                        target = torch.empty(p.orig_shape, dtype=p.dtype, device=self.device)
                        target.copy_(p.p_cpu, non_blocking=False)
                        p.data = target
                    elif p.data.device != self.device:
                        p.data = p.data.to(self.device, non_blocking=False)

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()



