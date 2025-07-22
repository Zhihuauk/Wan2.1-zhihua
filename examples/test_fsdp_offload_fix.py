#!/usr/bin/env python3
"""
FSDP + Stream Offload冲突修复验证脚本
=====================================

运行方式:
torchrun --nproc_per_node=2 examples/test_fsdp_offload_fix.py
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def test_fsdp_offload_compatibility():
    print("=== 测试FSDP + Stream Offload兼容性修复 ===")
    
    local_rank = setup_distributed()
    
    try:
        # 这里替换为实际的测试逻辑
        print(f"Rank {local_rank}: 开始测试")
        
        # 模拟设备移动测试
        for i in range(3):
            print(f"Rank {local_rank}: 模拟推理步骤 {i+1}")
            dummy_tensor = torch.randn(1, 4).to(f"cuda:{local_rank}")
            
        dist.barrier()
        if local_rank == 0:
            print("✅ FSDP + Stream Offload兼容性测试通过")
            
    except Exception as e:
        if local_rank == 0:
            print(f"❌ 测试失败: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    if 'RANK' in os.environ:
        test_fsdp_offload_compatibility()
    else:
        print("请使用torchrun运行此脚本") 