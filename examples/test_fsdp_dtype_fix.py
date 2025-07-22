#!/usr/bin/env python3
"""
FSDP数据类型修复验证脚本
======================

本脚本用于验证FSDP数据类型不一致问题的修复。

运行方式:
torchrun --nproc_per_node=2 examples/test_fsdp_dtype_fix.py
"""

import os
import torch
import torch.distributed as dist
from wan.vace import WanVace
from wan.configs.shared_config import SharedConfig


def setup_distributed():
    """设置分布式环境"""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_fsdp_dtype_fix():
    """测试FSDP数据类型修复"""
    print("=== 测试FSDP数据类型修复 ===")
    
    local_rank = setup_distributed()
    
    try:
        config = SharedConfig.from_file("wan/configs/wan_i2v_14B.py")
        checkpoint_dir = "your_checkpoint_path"  # 替换为实际路径
        
        if local_rank == 0:
            print("开始创建FSDP模型...")
            print(f"配置的参数类型: {getattr(config, 'param_dtype', 'not specified')}")
        
        # 创建模型（使用FSDP）
        model = WanVace(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=local_rank,
            dit_fsdp=True,   # 使用FSDP
            t5_fsdp=False,
            rank=local_rank
        )
        
        if local_rank == 0:
            print("✅ FSDP模型创建成功！")
            
            # 验证数据类型一致性
            print("\n=== 验证数据类型一致性 ===")
            
            # 检查不同部分的数据类型
            dtypes = set()
            
            # 检查blocks的数据类型
            if len(model.model.blocks) > 0:
                for param in model.model.blocks[0].parameters():
                    dtypes.add(param.dtype)
                    break
            
            # 检查vace_blocks的数据类型  
            if len(model.model.vace_blocks) > 0:
                for param in model.model.vace_blocks[0].parameters():
                    dtypes.add(param.dtype)
                    break
            
            # 检查其他模块的数据类型
            for name in ["patch_embedding", "time_embedding", "text_embedding"]:
                if hasattr(model.model, name):
                    module = getattr(model.model, name)
                    for param in module.parameters():
                        dtypes.add(param.dtype)
                        break
            
            print(f"发现的数据类型: {dtypes}")
            
            if len(dtypes) == 1:
                print(f"✅ 数据类型统一: {list(dtypes)[0]}")
            else:
                print(f"❌ 数据类型不统一: {dtypes}")
            
            # 验证OffloadManager是否正常工作
            print("\n=== 验证OffloadManager ===")
            stats = model.offloader.get_memory_stats()
            print(f"Offload状态: {stats['offload_enabled']}")
            print(f"分布式模式: {stats['distributed']}")
            print(f"use_orig_params优化: {stats.get('use_orig_params_optimized', False)}")
            
            block_info = model.offloader.get_block_info()
            for group_name, info in block_info.items():
                print(f"{group_name}: FSDP包装={info['fsdp_wrapped']}/{info['total_blocks']}")
        
        # 同步所有进程
        dist.barrier()
        
        if local_rank == 0:
            print("✅ FSDP数据类型修复验证完成")
            
    except Exception as e:
        if local_rank == 0:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    test_fsdp_dtype_fix() 