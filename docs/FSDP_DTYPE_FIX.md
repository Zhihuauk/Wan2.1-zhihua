# FSDP数据类型修复说明

## 问题描述

在FSDP模式下运行时遇到以下错误：
```
ValueError: Must flatten tensors with uniform dtype but got torch.float16 and torch.float32
```

## 问题原因

1. **数据类型不一致**：模型中部分参数是`float16`，部分是`float32`
2. **FSDP配置不匹配**：FSDP默认使用`bfloat16`，但代码中使用了`.half()`（即`float16`）
3. **转换时机错误**：在FSDP包装前没有统一所有参数的数据类型

## 解决方案

### 修改内容

在 `wan/vace.py` 中进行了以下修改：

#### 1. 统一数据类型处理逻辑
```python
# 检测是否使用FSDP
will_use_fsdp = dit_fsdp and dist.is_initialized()

if will_use_fsdp:
    # FSDP场景：将整个模型转换为与FSDP配置一致的数据类型
    target_dtype = self.config.param_dtype if hasattr(self.config, 'param_dtype') else torch.bfloat16
    self.model = self.model.to(dtype=target_dtype)
    print(f"[FSDP] Converted entire model to {target_dtype} for FSDP compatibility")
else:
    # 单卡场景：只转换非流式模块并移动到GPU
    for name in ["patch_embedding", "time_embedding", "text_embedding",
                "vace_patch_embedding", "time_projection", "head"]:
        setattr(self.model, name, getattr(self.model, name).to(device).half())
```

#### 2. 关键改进点

- **FSDP场景**：将整个模型转换为统一的数据类型（默认`bfloat16`）
- **单卡场景**：保持原有逻辑，只转换非流式模块为`float16`
- **配置匹配**：使用与FSDP配置一致的`param_dtype`
- **时机正确**：在FSDP包装**之前**完成数据类型统一

### 验证方法

运行测试脚本验证修复：
```bash
torchrun --nproc_per_node=2 examples/test_fsdp_dtype_fix.py
```

预期输出：
```
✅ FSDP模型创建成功！
✅ 数据类型统一: torch.bfloat16
✅ FSDP数据类型修复验证完成
```

## 技术细节

### FSDP数据类型要求

FSDP要求所有参数具有统一的数据类型才能进行flatten操作：
- `use_orig_params=True` 时仍需要类型一致
- 默认配置使用 `param_dtype=torch.bfloat16`
- 混合精度训练需要明确的类型配置

### 数据类型对应关系

| 方法 | 数据类型 | 说明 |
|------|----------|------|
| `.half()` | `torch.float16` | 半精度浮点 |
| `.to(dtype=torch.bfloat16)` | `torch.bfloat16` | Brain Float 16 |
| `.float()` | `torch.float32` | 单精度浮点 |

### 兼容性保证

- **向后兼容**：单卡模式保持原有行为
- **配置驱动**：根据config中的`param_dtype`自动选择类型
- **错误处理**：提供清晰的日志输出和错误信息

## 注意事项

1. **配置一致性**：确保`config.param_dtype`与FSDP配置匹配
2. **内存使用**：bfloat16相比float16有不同的精度特性
3. **设备兼容性**：确保目标设备支持所选的数据类型
4. **checkpoint兼容性**：数据类型转换可能影响checkpoint加载

## 故障排除

如果仍遇到数据类型相关错误：

1. **检查配置**：
   ```python
   print(f"Config param_dtype: {getattr(config, 'param_dtype', 'not set')}")
   ```

2. **验证模型类型**：
   ```python
   for name, param in model.named_parameters():
       print(f"{name}: {param.dtype}")
   ```

3. **确认FSDP配置**：
   检查 `wan/distributed/fsdp.py` 中的 `param_dtype` 设置 