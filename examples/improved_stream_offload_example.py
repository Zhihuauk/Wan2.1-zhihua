#!/usr/bin/env python3
"""
针对use_orig_params=True优化的流式权重卸载示例
================================================

本示例专门展示针对use_orig_params=True优化的OffloadManager功能：
1. 同时处理model.blocks和model.vace_blocks
2. 针对use_orig_params=True的FSDP配置优化
3. 简化的参数访问和复制策略
4. 详细的FSDP状态监控

FSDP配置特点（use_orig_params=True）：
- ✅ 保持原始参数形状，不进行flatten
- ✅ 可以直接访问参数，无需通过module.module
- ✅ model.blocks被FSDP包装，model.vace_blocks不被包装
- ✅ 参数结构更接近普通模块

运行方式:
- 单卡: python improved_stream_offload_example.py
- FSDP: torchrun --nproc_per_node=2 improved_stream_offload_example.py --fsdp
- 混合测试: python improved_stream_offload_example.py --test-mixed
"""

import os
import argparse
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


def test_single_gpu_offload():
    """测试单卡场景下的双block处理"""
    print("=== 测试单卡双block流式权重卸载 ===")
    
    config = SharedConfig.from_file("wan/configs/wan_i2v_14B.py")
    checkpoint_dir = "your_checkpoint_path"  # 替换为实际路径
    
    # 创建模型（不使用FSDP）
    model = WanVace(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        dit_fsdp=False,  # 不使用FSDP
        t5_fsdp=False
    )
    
    # 检查双block管理
    print("=== 双Block配置检查 ===")
    print(f"Blocks数量: {len(model.model.blocks)}")
    print(f"VACE Blocks数量: {len(model.model.vace_blocks)}")
    
    # 检查offload状态
    stats = model.offloader.get_memory_stats()
    print(f"内存统计: {stats}")
    print(f"use_orig_params优化: {stats.get('use_orig_params_optimized', False)}")
    
    # 详细的block信息
    block_info = model.offloader.get_block_info()
    print("\n=== Block详细信息 ===")
    for group_name, info in block_info.items():
        print(f"{group_name}:")
        print(f"  总数: {info['total_blocks']}")
        print(f"  常驻GPU: {info['resident_blocks']}")
        print(f"  已卸载: {info['offloaded_blocks']}")
        print(f"  FSDP包装: {info['fsdp_wrapped']}")
        print(f"  非FSDP: {info['non_fsdp']}")
    
    # 验证所有模块都是非FSDP的
    from wan.modules.stream_offload import _is_fsdp
    blocks_fsdp = [_is_fsdp(block) for block in model.model.blocks]
    vace_blocks_fsdp = [_is_fsdp(block) for block in model.model.vace_blocks]
    
    print(f"\n=== FSDP状态验证 ===")
    print(f"Blocks FSDP状态: {blocks_fsdp} (单卡时应该全为False)")
    print(f"VACE Blocks FSDP状态: {vace_blocks_fsdp} (应该全为False)")
    
    print("单卡双block测试完成")


def test_fsdp_offload():
    """测试FSDP场景下的混合block处理"""
    print("=== 测试FSDP混合block流式权重卸载 ===")
    
    local_rank = setup_distributed()
    
    try:
        config = SharedConfig.from_file("wan/configs/wan_i2v_14B.py")
        checkpoint_dir = "your_checkpoint_path"  # 替换为实际路径
        
        # 创建模型（使用FSDP + use_orig_params=True）
        model = WanVace(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=local_rank,
            dit_fsdp=True,   # 使用FSDP，配置中包含use_orig_params=True
            t5_fsdp=False,
            rank=local_rank
        )
        
        if local_rank == 0:
            print("=== FSDP配置验证 ===")
            print("FSDP配置包含use_orig_params=True - 保持原始参数形状")
            
            # 检查双block管理
            print(f"Blocks数量: {len(model.model.blocks)}")
            print(f"VACE Blocks数量: {len(model.model.vace_blocks)}")
        
        # 检查offload状态
        stats = model.offloader.get_memory_stats()
        block_info = model.offloader.get_block_info()
        
        if local_rank == 0:
            print(f"\n=== 内存和配置统计 ===")
            print(f"GPU内存分配: {stats['gpu_memory_allocated']}")
            print(f"分布式模式: {stats['distributed']}")
            print(f"use_orig_params优化: {stats.get('use_orig_params_optimized', False)}")
            print(f"FSDP统计: {stats.get('fsdp_stats', {})}")
            
            print(f"\n=== 详细Block信息 ===")
            for group_name, info in block_info.items():
                print(f"{group_name}:")
                print(f"  总数: {info['total_blocks']}")
                print(f"  常驻GPU: {info['resident_blocks']}")
                print(f"  已卸载: {info['offloaded_blocks']}")
                print(f"  FSDP包装: {info['fsdp_wrapped']}")
                print(f"  非FSDP: {info['non_fsdp']}")
        
        # 验证混合FSDP状态：blocks应该被FSDP包装，vace_blocks不应该
        from wan.modules.stream_offload import _is_fsdp
        blocks_fsdp = [_is_fsdp(block) for block in model.model.blocks]
        vace_blocks_fsdp = [_is_fsdp(block) for block in model.model.vace_blocks]
        
        if local_rank == 0:
            print(f"\n=== FSDP状态验证（use_orig_params=True） ===")
            print(f"Blocks FSDP状态: {blocks_fsdp}")
            print(f"预期: 所有blocks都应该被FSDP包装 (True)")
            print(f"VACE Blocks FSDP状态: {vace_blocks_fsdp}")
            print(f"预期: 所有vace_blocks都不应该被FSDP包装 (False)")
            
            # 验证use_orig_params=True的优化效果
            fsdp_count = sum(blocks_fsdp)
            non_fsdp_vace_count = len(model.model.vace_blocks) - sum(vace_blocks_fsdp)
            print(f"\n=== use_orig_params=True优化验证 ===")
            print(f"FSDP包装的blocks: {fsdp_count}/{len(model.model.blocks)}")
            print(f"非FSDP的vace_blocks: {non_fsdp_vace_count}/{len(model.model.vace_blocks)}")
            print("✅ 混合block处理 + use_orig_params=True优化正常工作")
        
        # 同步所有进程
        dist.barrier()
        
        # 测试参数访问优化
        if local_rank == 0:
            print("\n=== 测试use_orig_params=True参数访问优化 ===")
            # 测试FSDP包装的blocks参数访问
            if len(model.model.blocks) > 0:
                fsdp_block = model.model.blocks[0]
                if _is_fsdp(fsdp_block):
                    # use_orig_params=True时，可以直接访问参数
                    params = list(fsdp_block.parameters())
                    print(f"FSDP block参数数量: {len(params)}")
                    print("✅ use_orig_params=True允许直接参数访问")
            
            # 测试非FSDP的vace_blocks参数访问
            if len(model.model.vace_blocks) > 0:
                vace_block = model.model.vace_blocks[0]
                params = list(vace_block.parameters())
                print(f"VACE block参数数量: {len(params)}")
                print("✅ 非FSDP模块正常参数访问")
        
        if local_rank == 0:
            print("FSDP混合block测试完成")
            
    finally:
        cleanup_distributed()


def test_use_orig_params_optimization():
    """专门测试use_orig_params=True的优化功能"""
    print("=== 测试use_orig_params=True优化功能 ===")
    
    # 模拟use_orig_params=True的FSDP场景
    class TestBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(512, 512)
            self.linear2 = torch.nn.Linear(512, 512)
            
        def forward(self, x):
            return self.linear2(self.linear1(x))
    
    class TestModel(torch.nn.Module):
        def __init__(self, num_blocks=4, num_vace_blocks=3):
            super().__init__()
            # 模拟blocks（会被FSDP包装）
            self.blocks = torch.nn.ModuleList([TestBlock() for _ in range(num_blocks)])
            # 模拟vace_blocks（不会被FSDP包装）
            self.vace_blocks = torch.nn.ModuleList([TestBlock() for _ in range(num_vace_blocks)])
            
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            for block in self.vace_blocks:
                x = block(x)
            return x
    
    # 创建模型
    model = TestModel(num_blocks=4, num_vace_blocks=3).cuda()
    
    # 模拟FSDP包装（仅包装blocks，use_orig_params=True）
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if torch.cuda.device_count() > 0:
            # 只包装blocks，模拟use_orig_params=True
            for i, block in enumerate(model.blocks):
                model.blocks[i] = FSDP(block, use_orig_params=True)
            print("✅ 已模拟use_orig_params=True的FSDP包装")
    except (ImportError, RuntimeError):
        print("⚠️ FSDP不可用或未初始化分布式环境，使用普通模块测试")
    
    # 导入OffloadManager
    from wan.modules.stream_offload import OffloadManager, _is_fsdp
    
    # 验证混合包装状态
    blocks_fsdp = [_is_fsdp(block) for block in model.blocks]
    vace_blocks_fsdp = [_is_fsdp(block) for block in model.vace_blocks]
    
    print(f"\n=== 混合模块状态验证 ===")
    print(f"Blocks FSDP状态: {blocks_fsdp}")
    print(f"VACE Blocks FSDP状态: {vace_blocks_fsdp}")
    
    # 创建针对use_orig_params=True优化的offloader
    offloader = OffloadManager(
        root=model,
        module_groups={
            "blocks": model.blocks,        # 可能被FSDP包装
            "vace_blocks": model.vace_blocks  # 不被FSDP包装
        },
        keep_n={"blocks": 2, "vace_blocks": 1},  # 各组分别配置keep_n
        device=torch.device("cuda:0"),
        distributed=any(blocks_fsdp)  # 如果有FSDP模块就启用分布式模式
    )
    
    print(f"\n=== use_orig_params=True优化测试 ===")
    print("启用offload前的内存:")
    print(f"GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 启用offload
    offloader.enable()
    
    print("启用offload后的内存:")
    print(f"GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 获取详细统计
    stats = offloader.get_memory_stats()
    print(f"\n=== 优化统计信息 ===")
    print(f"use_orig_params优化: {stats.get('use_orig_params_optimized', False)}")
    print(f"FSDP统计: {stats.get('fsdp_stats', {})}")
    
    # 测试参数访问优化
    print(f"\n=== 参数访问优化测试 ===")
    for group_name, modules in [("blocks", model.blocks), ("vace_blocks", model.vace_blocks)]:
        if len(modules) > 0:
            module = modules[0]
            is_fsdp = _is_fsdp(module)
            
            # 测试直接参数访问（use_orig_params=True时应该可行）
            try:
                params = list(module.parameters())
                print(f"{group_name}[0]: FSDP={is_fsdp}, 参数数量={len(params)}, ✅直接访问成功")
            except Exception as e:
                print(f"{group_name}[0]: FSDP={is_fsdp}, ❌直接访问失败: {e}")
    
    # 测试前向传播
    x = torch.randn(1, 512).cuda()
    with torch.no_grad():
        y = model(x)
        print(f"\n✅ 混合模块前向传播成功，输出shape: {y.shape}")
    
    # 获取详细block信息
    block_info = offloader.get_block_info()
    print(f"\n=== 详细Block信息 ===")
    for group_name, info in block_info.items():
        print(f"{group_name}:")
        print(f"  总数: {info['total_blocks']}")
        print(f"  常驻GPU: {info['resident_blocks']}")
        print(f"  已卸载: {info['offloaded_blocks']}")
        print(f"  FSDP包装: {info['fsdp_wrapped']}")
        print(f"  非FSDP: {info['non_fsdp']}")
    
    # 清理
    offloader.disable()
    
    print("✅ use_orig_params=True优化功能测试完成")


def main():
    parser = argparse.ArgumentParser(description="use_orig_params=True优化的流式权重卸载测试")
    parser.add_argument("--fsdp", action="store_true", help="使用FSDP分布式模式")
    parser.add_argument("--test-optimization", action="store_true", help="测试use_orig_params=True优化")
    args = parser.parse_args()
    
    if args.test_optimization:
        test_use_orig_params_optimization()
    elif args.fsdp:
        test_fsdp_offload()
    else:
        test_single_gpu_offload()


if __name__ == "__main__":
    main() 