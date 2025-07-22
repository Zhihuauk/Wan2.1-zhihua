# Stream Offload Manager - 使用指南

Stream Offload Manager 是一个内存优化工具，通过动态管理模型参数在CPU和GPU之间的传输，显著减少GPU内存使用量。本实现完全兼容现有的WanVace工作流，同时支持单卡和FSDP分布式场景。

## 特性

- **流式权重管理**: 动态加载/卸载模型参数
- **FSDP支持**: 完全兼容PyTorch FSDP
- **双模型支持**: 支持WanModel和VaceWanModel
- **可配置内存占用**: 可调节保留在GPU上的block数量
- **API友好**: 简单的启用/禁用接口
- **内存监控**: 实时GPU内存使用统计
- **上下文管理器**: 支持Python `with`语句

## 基本使用

### 1. 初始化模型（保持原有方式不变）

```python
from wan.vace import WanVace
from wan.configs import wan_vace_14B

# 现有的初始化方式完全不变
config = wan_vace_14B()
model = WanVace(
    config=config,
    checkpoint_dir="/path/to/checkpoints",
    dit_fsdp=True  # 启用FSDP
)
```

### 2. 手动控制Offload

```python
# 启用offload，保留2个blocks在GPU上
model.enable_offload(keep_n=2)

# 查看内存使用情况
stats = model.get_memory_stats()
print(f"GPU Memory: {stats['gpu_memory_allocated']}")
print(f"Offloaded blocks: {stats['offloaded_blocks']}")

# 生成视频（offload自动工作）
result = model.generate(
    input_prompt="A beautiful landscape",
    input_frames=input_frames,
    input_masks=input_masks,
    input_ref_images=input_ref_images,
    size=(1280, 720),
    frame_num=81
)

# 禁用offload
model.disable_offload()
```

### 3. 使用上下文管理器

```python
# 临时启用offload
with model.offload_context(keep_n=1):
    result = model.generate(...)
# 自动禁用offload
```

### 4. 配置文件控制（推荐）

在配置文件中设置自动启用：

```python
# 在config中添加
config.offload_keep_n = 2        # 保留2个blocks在GPU
config.auto_enable_offload = True  # 自动启用offload
```

这样模型初始化时会自动启用offload，无需额外代码。

## API参考

### 核心方法

#### `enable_offload(keep_n=None)`
启用流式权重offload。

- `keep_n`: 保留在GPU上的blocks数量

#### `disable_offload()`
禁用offload，恢复所有参数到GPU。

#### `get_memory_stats()`
获取详细的内存使用统计。

```python
stats = model.get_memory_stats()
# 返回:
# {
#     "gpu_memory_allocated": "12.34 GB",
#     "gpu_memory_reserved": "13.56 GB", 
#     "offload_enabled": True,
#     "total_blocks": 32,
#     "resident_blocks": 2,
#     "offloaded_blocks": 30,
#     "keep_n_config": {"blocks": 1, "vace_blocks": 1},
#     "device": "cuda:0",
#     "distributed": True
# }
```

#### `get_offload_info()`
获取offload配置详情。

#### `is_offload_enabled()`
检查offload是否已启用。

#### `set_offload_keep_n(keep_n, auto_restart=True)`
动态调整keep_n设置。

#### `offload_context(keep_n=None)`
返回上下文管理器。

## 配置建议

### 内存vs性能权衡

| keep_n | 内存使用 | 推理速度 | 适用场景 |
|--------|----------|----------|----------|
| 1      | 最低     | 较慢     | 极限内存约束 |
| 2      | 低       | 中等     | 平衡选择（推荐） |
| 4      | 中       | 快       | 大多数情况 |
| 8+     | 高       | 最快     | 充足GPU内存 |

### FSDP场景

```python
# FSDP自动检测和启用
model = WanVace(config, checkpoint_dir, dit_fsdp=True)

# 检查manager类型
info = model.get_offload_info()
print(f"Manager type: {info['manager_type']}")  # "FSDP" 或 "Single-GPU"
```

## 内存监控示例

```python
def monitor_generation():
    # 启用前
    stats_before = model.get_memory_stats()
    
    # 启用offload
    model.enable_offload(keep_n=2)
    stats_after = model.get_memory_stats()
    
    print(f"Memory saved: {stats_before['gpu_memory_allocated']} -> {stats_after['gpu_memory_allocated']}")
    
    # 生成过程中监控
    for step in range(5):
        model.enable_offload(keep_n=step+1)
        stats = model.get_memory_stats()
        print(f"keep_n={step+1}: {stats['gpu_memory_allocated']}")

monitor_generation()
```

## 错误处理

```python
try:
    model.enable_offload(keep_n=2)
    result = model.generate(...)
except Exception as e:
    print(f"Generation failed: {e}")
finally:
    # 确保正确清理
    if model.is_offload_enabled():
        model.disable_offload()
```

## 与现有代码的兼容性

### 完全向后兼容

```python
# 原有代码无需修改
model = WanVace(config, checkpoint_dir, dit_fsdp=True)
result = model.generate(...)  # 正常工作

# 需要节省内存时，只需添加两行
model.enable_offload(keep_n=2)   # 添加这行
result = model.generate(...)      # 无需修改
model.disable_offload()           # 添加这行（可选）
```

### 配置文件升级

```python
# 旧配置保持不变
config = wan_vace_14B()

# 新配置（可选）
config.offload_keep_n = 2
config.auto_enable_offload = True
```

## 性能优化建议

1. **预热**: 第一次推理可能较慢，后续会更快
2. **批处理**: 连续推理时保持offload启用
3. **内存监控**: 定期检查内存使用，调整keep_n
4. **FSDP优化**: 在分布式场景中，offload效果更显著

## 故障排查

### 常见问题

1. **CUDA OOM**: 减少keep_n值
2. **推理变慢**: 增加keep_n值或禁用offload
3. **初始化失败**: 检查FSDP配置

### 调试信息

```python
# 检查offload状态
print(f"Offload enabled: {model.is_offload_enabled()}")

# 查看详细信息
info = model.get_offload_info()
for key, value in info.items():
    print(f"{key}: {value}")

# 内存统计
stats = model.get_memory_stats()
print(f"Current memory: {stats['gpu_memory_allocated']}")
```

## 更新日志

- 增强现有的OffloadManager，增加便利方法
- 保持原有调用方式完全不变
- 添加内存监控和统计功能
- 支持上下文管理器
- 改进错误处理和日志输出
- 完善API文档和使用示例 