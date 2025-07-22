# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn


class OffloadManager:
    """
    Manages streaming offload of model parameters for FSDP scenarios.
    Supports multiple module types (e.g., blocks and vace_blocks in WanVace model).
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize the OffloadManager.
        
        Args:
            device_id (int): Target GPU device ID
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.device_id = device_id
        
        # Streams for async memory operations
        self.eval_h2d_stream = None  # Host to Device stream
        self.eval_d2h_stream = None  # Device to Host stream
        
        # Module configurations
        self.module_configs = {}  # {module_name: {'modules': modules, 'resident_count': count}}
        self.module_hooks = {}    # {module_name: {'pre_hooks': [], 'post_hooks': []}}
        
        self._initialized = False
    
    def register_modules(
        self, 
        module_name: str, 
        modules: Union[nn.ModuleList, List[nn.Module]], 
        resident_count: int = 2
    ):
        """
        Register a set of modules for offload management.
        
        Args:
            module_name (str): Name identifier for this module set (e.g., 'blocks', 'vace_blocks')
            modules (Union[nn.ModuleList, List[nn.Module]]): List of modules to manage
            resident_count (int): Number of modules to keep resident on GPU
        """
        if self._initialized:
            raise RuntimeError("Cannot register modules after initialization. Call this before enable_offload().")
        
        self.module_configs[module_name] = {
            'modules': modules,
            'resident_count': resident_count,
            'depth': len(modules)
        }
        
        logging.info(f"Registered {len(modules)} modules for '{module_name}' with {resident_count} resident blocks")
    
    def enable_offload(self):
        """
        Enable streaming offload for all registered modules.
        This should be called after all modules are registered.
        """
        if self._initialized:
            logging.warning("Offload already enabled, skipping...")
            return
        
        if not self.module_configs:
            raise RuntimeError("No modules registered. Call register_modules() first.")
        
        # Initialize CUDA streams
        self.eval_h2d_stream = torch.cuda.Stream()
        self.eval_d2h_stream = torch.cuda.Stream()
        
        # Process each module type
        for module_name, config in self.module_configs.items():
            self._setup_module_offload(module_name, config)
        
        self._initialized = True
        logging.info("Streaming offload enabled for all registered modules")
    
    def _setup_module_offload(self, module_name: str, config: Dict):
        """
        Setup offload for a specific module type.
        
        Args:
            module_name (str): Name of the module set
            config (Dict): Configuration for this module set
        """
        modules = config['modules']
        resident_count = config['resident_count']
        depth = config['depth']
        
        # Store hooks for this module type
        self.module_hooks[module_name] = {'pre_hooks': [], 'post_hooks': []}
        
        # Parameter to host warmup
        logging.info(f"Moving non-resident parameters to CPU for '{module_name}'...")
        for blk_idx in range(resident_count, depth):
            for p in modules[blk_idx].parameters():
                with torch.cuda.stream(self.eval_d2h_stream):
                    if not hasattr(p, "p_cpu"):
                        p_cpu = torch.empty(p.data.shape, dtype=p.dtype, pin_memory=True, device='cpu')
                        setattr(p, "p_cpu", p_cpu)
                    
                    is_slice_tensor = p.data.untyped_storage().size() != p.data.numel()
                    storage_size = p.data.untyped_storage().size()
                    
                    if is_slice_tensor:
                        p.p_cpu.copy_(p.data, non_blocking=True)
                    else:
                        p.p_cpu.untyped_storage().copy_(p.data.untyped_storage(), non_blocking=True)
                    
                    setattr(p, "storage_size", storage_size)
                    setattr(p, "is_slice_tensor", is_slice_tensor)
        
        # Wait for copy operations to complete
        torch.cuda.current_stream().wait_stream(self.eval_d2h_stream)
        torch.cuda.default_stream().wait_stream(self.eval_d2h_stream)
        
        # Resize non-resident parameters to free GPU memory
        for blk_idx in range(resident_count, depth):
            for p in modules[blk_idx].parameters():
                p.data.untyped_storage().resize_(0)
        
        # Register hooks for each module
        for blk_idx, blk in enumerate(modules):
            # Create closure to capture current values
            pre_hook = self._create_pre_hook(module_name, blk_idx, resident_count, depth, modules)
            post_hook = self._create_post_hook(module_name, blk_idx, resident_count, modules)
            
            # Register hooks
            pre_handle = blk.register_forward_pre_hook(pre_hook)
            self.module_hooks[module_name]['pre_hooks'].append(pre_handle)
            
            if blk_idx >= resident_count:
                post_handle = blk.register_forward_hook(post_hook)
                self.module_hooks[module_name]['post_hooks'].append(post_handle)
        
        logging.info(f"Setup completed for '{module_name}': {resident_count}/{depth} blocks resident")
    
    def _create_pre_hook(self, module_name: str, blk_idx: int, resident_count: int, depth: int, modules: List[nn.Module]):
        """
        Create a pre-forward hook for parameter loading.
        """
        def parameter_to_device_hook(module, input):
            to_device_index = blk_idx + resident_count
            forward_event = torch.cuda.Event()
            forward_event.record()
            
            if to_device_index < depth:
                with torch.cuda.stream(self.eval_h2d_stream):
                    for p in modules[to_device_index].parameters():
                        p.data.untyped_storage().resize_(p.storage_size)
                        self.eval_h2d_stream.wait_event(forward_event)
                        
                        if p.is_slice_tensor:
                            p.data.copy_(p.p_cpu, non_blocking=True)
                        else:
                            p.data.untyped_storage().copy_(p.p_cpu.untyped_storage(), non_blocking=True)
        
        return parameter_to_device_hook
    
    def _create_post_hook(self, module_name: str, blk_idx: int, resident_count: int, modules: List[nn.Module]):
        """
        Create a post-forward hook for parameter cleanup.
        """
        def parameter_to_resize_hook(module, input, output):
            to_resize_index = blk_idx
            
            if to_resize_index >= resident_count:
                for p in modules[to_resize_index].parameters():
                    p.data.untyped_storage().resize_(0)
            
            torch.cuda.current_stream().wait_stream(self.eval_h2d_stream)
            torch.cuda.default_stream().wait_stream(self.eval_h2d_stream)
        
        return parameter_to_resize_hook
    
    def disable_offload(self):
        """
        Disable streaming offload and remove all hooks.
        """
        if not self._initialized:
            logging.warning("Offload not enabled, nothing to disable")
            return
        
        # Remove all hooks
        for module_name, hooks in self.module_hooks.items():
            for handle in hooks['pre_hooks']:
                handle.remove()
            for handle in hooks['post_hooks']:
                handle.remove()
        
        # Clear hook storage
        self.module_hooks.clear()
        
        # Restore all parameters to GPU
        for module_name, config in self.module_configs.items():
            self._restore_module_parameters(module_name, config)
        
        self._initialized = False
        logging.info("Streaming offload disabled")
    
    def _restore_module_parameters(self, module_name: str, config: Dict):
        """
        Restore all parameters of a module type back to GPU.
        """
        modules = config['modules']
        resident_count = config['resident_count']
        depth = config['depth']
        
        for blk_idx in range(resident_count, depth):
            for p in modules[blk_idx].parameters():
                if hasattr(p, 'p_cpu') and hasattr(p, 'storage_size'):
                    p.data.untyped_storage().resize_(p.storage_size)
                    if hasattr(p, 'is_slice_tensor') and p.is_slice_tensor:
                        p.data.copy_(p.p_cpu)
                    else:
                        p.data.untyped_storage().copy_(p.p_cpu.untyped_storage())
                    
                    # Clean up attributes
                    delattr(p, 'p_cpu')
                    delattr(p, 'storage_size')
                    delattr(p, 'is_slice_tensor')
    
    def get_memory_stats(self) -> Dict[str, Dict]:
        """
        Get memory statistics for each registered module type.
        """
        stats = {}
        for module_name, config in self.module_configs.items():
            modules = config['modules']
            resident_count = config['resident_count']
            depth = config['depth']
            
            resident_params = sum(p.numel() for i in range(min(resident_count, depth)) for p in modules[i].parameters())
            offloaded_params = sum(p.numel() for i in range(resident_count, depth) for p in modules[i].parameters())
            
            stats[module_name] = {
                'total_blocks': depth,
                'resident_blocks': resident_count,
                'offloaded_blocks': depth - resident_count,
                'resident_parameters': resident_params,
                'offloaded_parameters': offloaded_params,
                'total_parameters': resident_params + offloaded_params
            }
        
        return stats
    
    def __del__(self):
        """
        Cleanup when object is destroyed.
        """
        if self._initialized:
            try:
                self.disable_offload()
            except:
                pass  # Ignore errors during cleanup 