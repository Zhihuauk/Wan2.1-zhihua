import torch
import weakref
import logging
from typing import Dict, Sequence, Optional

# Utility functions to ensure that the module is FSDP or DDP
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

def _get_fsdp_parameters(module):
    """
    获取FSDP模块的参数，针对use_orig_params=True进行优化。
    当use_orig_params=True时，FSDP保持原始参数形状，可以直接访问参数。
    """
    if _is_fsdp(module):
        # 对于use_orig_params=True的FSDP，直接访问参数即可
        # 不需要通过module.module访问
        return module.parameters(recurse=True)
    elif _is_ddp(module):
        return module.module.parameters(recurse=True)
    else:
        return module.parameters(recurse=True)

def _get_fsdp_buffers(module):
    """
    获取FSDP模块的缓冲区，针对use_orig_params=True进行优化。
    """
    if _is_fsdp(module):
        # 对于use_orig_params=True的FSDP，直接访问buffers即可
        return module.buffers(recurse=True)
    elif _is_ddp(module):
        return module.module.buffers(recurse=True)
    else:
        return module.buffers(recurse=True)

class OffloadManager:
    """
    通用流式权重管理器，支持单卡和分布式（FSDP/DDP）推理。
    针对use_orig_params=True的FSDP配置进行优化，同时处理blocks和vace_blocks。
    """

    def __init__(
        self,
        root: torch.nn.Module, 
        module_groups: Dict[str, Sequence[torch.nn.Module]],
        keep_n: Dict[str, int] | int,
        device: Optional[torch.device] = None,
        distributed: bool = False,
    ):
        self.root = weakref.proxy(root) #weakref 避免循环引用
        self.device = device or next(root.parameters()).device
        self.keep_n = ( #构建一个 keep_n 字典
            {k: keep_n for k in module_groups} if isinstance(keep_n, int) else keep_n)
        
        self.groups = module_groups #modules 的分组
        self.distributed = distributed #处理分布式推理

        self.h2d_stream = torch.cuda.Stream() #host to device stream
        self.d2h_stream = torch.cuda.Stream() #device to host stream
        self.enabled = False #是否启用流式 offload
        self.handles = [] # hooks 列表，统一保存已注册 hook，便于关闭

        # 递归注册 index/depth
        for name, grp in self.groups.items():
            for i, m in enumerate(grp):
                setattr(m, "_stream_offload_index", i) #注册属性 第几层 _stream_offload_index
                setattr(m, "_stream_offload_depth", len(grp)) #注册属性 多深 _stream_offload_depth
                setattr(m, "_stream_offload_group", name) #注册属性 所属组名

    def enable(self):
        if self.enabled:
            return
        # no CUDA check
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for streaming-offload.")

        # 0. 保证前 keep_n 层在 self.device
        for tag, grp in self.groups.items():
            stay_on_gpu = min(self.keep_n[tag], len(grp)) #取输入层数和组长度的最小值
            for i in range(stay_on_gpu): #将前 stay_on_gpu 层移动到目标设备，gpu/npu
                self._move_module_to_device(grp[i], self.device)

        # 1. keep_n 之后的层做 offload
        for tag, grp in self.groups.items():
            if len(grp) > self.keep_n[tag]: #  对于大于 keep_n 的层，进行 offload
                self._setup_parameter_offload(grp, self.keep_n[tag])

        # 2. 注册 hooks
        for tag, grp in self.groups.items(): #遍历所有的 groups
            for m in grp: #遍历所有的模块，并注册pre_hooks 和 post_hooks
                pre_h = m.register_forward_pre_hook( #这里重点，每次forward 前，先将 keep_n 之后的层预加载到 GPU
                    self._prefetch_hook_factory(tag), with_kwargs=False
                )
                self.handles.append(pre_h)
                if getattr(m, "_stream_offload_index") >= self.keep_n[tag]: #keepn之外的层，需要每轮都释放
                    post_h = m.register_forward_hook( #这个接口发生在，forward后，释放 keep_n 之前的层
                        self._release_hook_factory(tag), with_kwargs=False
                    )
                    self.handles.append(post_h)

        self.enabled = True
        #输出 offload 状态
        print(f"[stream_offload] enabled (keep_n={self.keep_n}, device={self.device}, distributed={self.distributed})")
        
        # 输出各组的详细信息
        for tag, grp in self.groups.items():
            fsdp_wrapped = sum(1 for m in grp if _is_fsdp(m))
            print(f"[stream_offload] {tag}: {len(grp)} modules, {fsdp_wrapped} FSDP-wrapped, keep_n={self.keep_n[tag]}")

    # 取消启用流式 offload
    def disable(self):
        if not self.enabled:
            return
        self._restore_all()
        self._remove_hooks()
        self.enabled = False
        print("[stream_offload] disabled – all weights back on GPU")

    # --- internal ---
    # 将模块移动到指定设备
    def _move_module_to_device(self, module, device):
        # 对于use_orig_params=True的FSDP，直接移动即可
        # 不需要特殊的wrapper处理
        if _is_fsdp(module):
            # use_orig_params=True时，可以直接to(device)
            module.to(device, non_blocking=False)
        elif _is_ddp(module):
            module.module.to(device, non_blocking=False)
        else:
            module.to(device, non_blocking=False)

    @staticmethod
    #遍历一个 Module 内"真正占显存的张量，针对use_orig_params=True优化
    def _iter_tensors(m: torch.nn.Module):
        # 改进：针对use_orig_params=True，简化参数访问
        # use_orig_params=True时，FSDP模块的参数访问方式和普通模块类似
        if _is_fsdp(m):
            # use_orig_params=True时，直接访问参数
            yield from m.parameters(recurse=True)
            yield from m.buffers(recurse=True)
        elif _is_ddp(m):
            # DDP仍需要通过module访问
            yield from m.module.parameters(recurse=True)
            yield from m.module.buffers(recurse=True)
        else:
            # 普通模块直接访问
            yield from m.parameters(recurse=True)
            yield from m.buffers(recurse=True)

    def _prefetch_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        grp = self.groups[tag]

        def hook(module, _inputs):
            idx = getattr(module, "_stream_offload_index") # 当前层序号
            if idx + keep_n >= len(grp): # 尾部几层无需预取
                return
            nxt = grp[idx + keep_n] # 目标：窗口后一层
            evt = torch.cuda.Event(); evt.record() # 在"当前计算流"打标记
            with torch.cuda.stream(self.h2d_stream): # 切换到 H2D Stream
                self.h2d_stream.wait_event(evt) # 等计算流走到 evt
                for p in self._iter_tensors(nxt): # 逐张量
                    self._ensure_gpu_tensor(p)
            # torch.cuda.current_stream().wait_stream(self.h2d_stream)
        return hook

    # 对"滑动区"层（序号 ≥ keep_n）生效；窗口内的常驻层始终留在 GPU
    def _release_hook_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        def hook(module, _inp, _out):
            idx = getattr(module, "_stream_offload_index")
            if idx < keep_n: # 常驻窗口内的层不释放
                return
            for p in self._iter_tensors(module):
                self._release_tensor(p) #把实际显存归还
        return hook

   #只对 非常驻层 做首轮 offload，把整个拷贝过程放在 d2h_stream，避免阻塞主计算流

    def _setup_parameter_offload(self, grp, keep_n):
        for idx in range(keep_n, len(grp)):
            blk = grp[idx]
            with torch.cuda.stream(self.d2h_stream):
                for p in self._iter_tensors(blk):
                    if not hasattr(p, "_stream_offload_p_cpu"):
                        # 针对use_orig_params=True优化：简化slice tensor检测
                        # use_orig_params=True时参数形状更稳定
                        is_slice_tensor = p.data.untyped_storage().size() != p.data.numel()
                        storage_size = p.data.untyped_storage().size()
                        
                        # 只保存本 rank 的 shard
                        p._stream_offload_p_cpu = torch.empty_like(p.data, pin_memory=True, device="cpu")  
                        p._stream_offload_orig_shape = tuple(p.shape)
                        p._stream_offload_storage_size = storage_size
                        p._stream_offload_is_slice_tensor = is_slice_tensor
                        
                    # 针对use_orig_params=True优化复制策略
                    if p._stream_offload_is_slice_tensor:
                        p._stream_offload_p_cpu.copy_(p.data, non_blocking=True)
                    else:
                        # use_orig_params=True时，优先使用tensor级别的copy
                        # 因为参数结构更接近普通模块
                        try:
                            p._stream_offload_p_cpu.copy_(p.data, non_blocking=True)
                        except:
                            # 极端情况下的退化处理
                            p._stream_offload_p_cpu.untyped_storage().copy_(p.data.untyped_storage(), non_blocking=True)
                            
            torch.cuda.current_stream().wait_stream(self.d2h_stream) # 等复制结束
            
        # 立即释放 GPU storage
        for idx in range(keep_n, len(grp)):
            blk = grp[idx]
            for p in self._iter_tensors(blk):
                self._release_tensor(p) #调 _release_tensor() 把 GPU 显存真正腾出


    #需要张量时保证它在 GPU 且可用，针对use_orig_params=True优化
    def _ensure_gpu_tensor(self, p: torch.Tensor):
        dev = self.device 
        need_alloc = ( #  	三种情形需要重新申请 GPU storage
            p.data.device != dev or  # a. 不在目标 GPU
            p.data.untyped_storage().size() == 0 or # b. storage 被 resize_(0)
            p.data.untyped_storage().size() < getattr(p, "_stream_offload_storage_size", p.numel()) # c. storage 太小
        )
        if need_alloc:
            target_shape = getattr(p, "_stream_offload_orig_shape", tuple(  # 取原始 shape
                p._stream_offload_p_cpu.shape if hasattr(p, "_stream_offload_p_cpu") else p.shape)
            )
            # 针对use_orig_params=True：简化分配逻辑
            if hasattr(p, "_stream_offload_storage_size"):
                # 先分配tensor
                p.data = torch.empty(target_shape, dtype=p.dtype, device=dev)
                # 对于use_orig_params=True，通常不需要复杂的storage resize
                try:
                    if p._stream_offload_storage_size != p.data.untyped_storage().size():
                        p.data.untyped_storage().resize_(p._stream_offload_storage_size)
                except:
                    # 如果resize失败，保持tensor分配即可
                    pass
            else:
                p.data = torch.empty(target_shape, dtype=p.dtype, device=dev) # 重新分配到目标设备
                
        # 针对use_orig_params=True优化恢复策略
        if hasattr(p, "_stream_offload_p_cpu"):
            # use_orig_params=True时，优先使用tensor copy
            try:
                p.data.copy_(p._stream_offload_p_cpu, non_blocking=True)
            except:
                # 退化到storage copy
                if not getattr(p, "_stream_offload_is_slice_tensor", False):
                    try:
                        p.data.untyped_storage().copy_(p._stream_offload_p_cpu.untyped_storage(), non_blocking=True)
                    except:
                        pass  # 如果都失败，保持原状
        else:
            if need_alloc or p.data.device != dev:
                p.data = p.data.to(dev, non_blocking=True)
        #  清除"已释放"标记
        if getattr(p, "_stream_offload_released", False):
            p._stream_offload_released = False

    @staticmethod
    def _release_tensor(p: torch.Tensor):
        try:
            p.data.untyped_storage().resize_(0) #a. 先缩 storage，resize_(0) 在 C++ 侧直接归还 CUDA memory，但 保持 Tensor 对象不变 → 计算图里的引用仍安全
            p.data.resize_(0)   #   b. 再把 shape 设为 0
            p._stream_offload_released = True
        except Exception: #  极端情况下退化，某些特殊 Tensor（e.g. view, tensor subclass）untyped_storage() 不支持 resize_，就用 torch.empty(0) 强行替换数据指针
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
            p._stream_offload_released = True

    
    #在 disable() 时调用，把所有 offload 权重完整搬回 GPU，确保模型后续还能常规使用或保存
    def _restore_all(self):
        for grp in self.groups.values():
            for m in grp:
                for p in self._iter_tensors(m):
                    if hasattr(p, "_stream_offload_p_cpu"):  # 仅处理曾 offload 过的
                        if p.data.untyped_storage().size() == 0:
                            target = torch.empty(p._stream_offload_orig_shape, dtype=p.dtype, device=self.device)
                            
                            # 针对use_orig_params=True优化恢复
                            if hasattr(p, "_stream_offload_storage_size"):
                                try:
                                    if p._stream_offload_storage_size != target.untyped_storage().size():
                                        target.untyped_storage().resize_(p._stream_offload_storage_size)
                                except:
                                    pass
                                    
                            # use_orig_params=True时优先tensor copy
                            try:
                                target.copy_(p._stream_offload_p_cpu, non_blocking=False)
                            except:
                                # 退化到storage copy
                                if not getattr(p, "_stream_offload_is_slice_tensor", False):
                                    try:
                                        target.untyped_storage().copy_(p._stream_offload_p_cpu.untyped_storage(), non_blocking=False)
                                    except:
                                        pass  # 保持原状
                            p.data = target
                        elif p.data.device != self.device:
                            p.data = p.data.to(self.device, non_blocking=False)
    #torch 的 hook 只有显式 handle.remove() 才会解绑；否则 Python 对象虽然要析构，但 C++ 端 lambda 依旧留在 Module._forward_hooks 字典里。
    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # ============== 新增便利方法 ==============
    
    def get_memory_stats(self):
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {"gpu_memory": "N/A"}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
        
        # 统计模型结构信息
        total_blocks = sum(len(grp) for grp in self.groups.values())
        resident_blocks = sum(self.keep_n.values())
        
        # 统计FSDP包装情况
        fsdp_stats = {}
        for group_name, grp in self.groups.items():
            fsdp_wrapped = sum(1 for m in grp if _is_fsdp(m))
            fsdp_stats[f"{group_name}_fsdp_wrapped"] = fsdp_wrapped
            fsdp_stats[f"{group_name}_total"] = len(grp)
        
        return {
            "gpu_memory_allocated": f"{allocated:.2f} GB",
            "gpu_memory_reserved": f"{reserved:.2f} GB",
            "offload_enabled": self.enabled,
            "total_blocks": total_blocks,
            "resident_blocks": resident_blocks,
            "offloaded_blocks": total_blocks - resident_blocks,
            "keep_n_config": dict(self.keep_n),
            "device": str(self.device),
            "distributed": self.distributed,
            "fsdp_stats": fsdp_stats,
            "use_orig_params_optimized": True  # 标记已针对use_orig_params优化
        }
    
    def set_keep_n(self, new_keep_n: Dict[str, int] | int):
        """
        动态调整keep_n配置。需要先disable再enable才能生效。
        
        Args:
            new_keep_n: 新的keep_n配置
        """
        if isinstance(new_keep_n, int):
            self.keep_n = {k: new_keep_n for k in self.groups}
        else:
            self.keep_n.update(new_keep_n)
        
        if self.enabled:
            logging.warning("keep_n changed while offload enabled. Call disable() then enable() to apply changes.")
    
    def is_enabled(self):
        """Check if offload is currently enabled."""
        return self.enabled
    
    def get_block_info(self):
        """Get detailed information about blocks."""
        info = {}
        for group_name, blocks in self.groups.items():
            fsdp_wrapped = sum(1 for m in blocks if _is_fsdp(m))
            info[group_name] = {
                "total_blocks": len(blocks),
                "resident_blocks": self.keep_n[group_name],
                "offloaded_blocks": max(0, len(blocks) - self.keep_n[group_name]),
                "fsdp_wrapped": fsdp_wrapped,
                "non_fsdp": len(blocks) - fsdp_wrapped
            }
        return info
    
    # 上下文管理器支持
    def __enter__(self):
        """Context manager entry."""
        if not self.enabled:
            self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.enabled:
            self.disable()
