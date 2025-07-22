# Streaming Offload for Memory Optimization

## Overview

The streaming offload feature is designed to reduce GPU memory usage in FSDP (Fully Sharded Data Parallel) scenarios by dynamically managing model parameters. Instead of keeping all model blocks resident on GPU, this feature keeps only a subset of blocks on GPU and streams the rest from CPU memory as needed during inference.

## Key Features

- **Memory Efficient**: Reduces GPU memory usage by offloading non-critical parameters to CPU
- **Transparent**: Works seamlessly with existing WanVace models without changing inference API
- **Configurable**: Allows fine-tuning of which and how many blocks to keep resident on GPU
- **Multi-block Support**: Handles both main transformer blocks and VACE-specific blocks
- **Multiprocessing Compatible**: Works with both single-process and multi-process model instances

## How It Works

1. **Initialization**: During model initialization, parameters from non-resident blocks are copied to pinned CPU memory
2. **Dynamic Loading**: Forward hooks automatically load required parameters from CPU to GPU just before they're needed
3. **Memory Cleanup**: Post-forward hooks immediately free GPU memory after block execution
4. **Asynchronous Operations**: Uses CUDA streams for non-blocking memory transfers

## Usage

### Basic Usage

```python
from wan.vace import WanVace
from wan.configs.wan_t2v_14B import config

# Initialize with streaming offload enabled
model = WanVace(
    config=config,
    checkpoint_dir="path/to/checkpoint",
    device_id=0,
    enable_offload=True,           # Enable streaming offload
    blocks_resident_count=2,       # Keep 2 main blocks on GPU
    vace_blocks_resident_count=2   # Keep 2 vace blocks on GPU
)

# Check if offload is active
print(f"Offload enabled: {model.is_offload_enabled()}")

# Get memory statistics
stats = model.get_offload_memory_stats()
for module_name, module_stats in stats.items():
    print(f"{module_name}: {module_stats['resident_blocks']}/{module_stats['total_blocks']} blocks resident")
```

### Multiprocessing Usage

```python
from wan.vace import WanVaceMP

# Initialize multiprocessing version with offload
model_mp = WanVaceMP(
    config=config,
    checkpoint_dir="path/to/checkpoint",
    enable_offload=True,
    blocks_resident_count=2,
    vace_blocks_resident_count=2
)

# Each worker process will automatically have offload enabled
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_offload` | bool | False | Enable/disable streaming offload |
| `blocks_resident_count` | int | 2 | Number of main transformer blocks to keep on GPU |
| `vace_blocks_resident_count` | int | 2 | Number of VACE blocks to keep on GPU |

## Memory Trade-offs

### Benefits
- **Reduced GPU Memory**: Significant reduction in GPU memory usage
- **Larger Model Support**: Enables running larger models on limited GPU memory
- **Flexible Configuration**: Fine-tune memory vs. performance trade-offs

### Considerations
- **Performance Impact**: Additional latency from CPU-GPU memory transfers
- **CPU Memory Usage**: Increased CPU memory usage for offloaded parameters
- **Transfer Overhead**: CUDA stream synchronization overhead

## Performance Recommendations

1. **Resident Block Count**: Start with 2-4 resident blocks per module type and adjust based on your memory constraints
2. **Memory Bandwidth**: Ensure sufficient CPU-GPU memory bandwidth for smooth transfers
3. **Pinned Memory**: The implementation uses pinned memory for faster transfers
4. **Batch Size**: Consider reducing batch size to maximize benefits

## Implementation Details

### OffloadManager Class

The core functionality is implemented in the `OffloadManager` class located in `wan/modules/stream_offload.py`. Key features:

- **Module Registration**: Supports multiple module types (blocks, vace_blocks, etc.)
- **Hook Management**: Automatically manages forward pre/post hooks
- **Memory Tracking**: Provides detailed memory usage statistics
- **Stream Management**: Uses dedicated CUDA streams for async operations

### Hook Mechanism

1. **Pre-forward Hook**: Triggered before block execution
   - Loads parameters from CPU to GPU
   - Uses CUDA events for synchronization
   - Handles both slice tensors and regular tensors

2. **Post-forward Hook**: Triggered after block execution
   - Frees GPU memory by resizing tensor storage to 0
   - Synchronizes CUDA streams

## Monitoring and Debugging

### Memory Statistics

```python
# Get detailed memory statistics
stats = model.get_offload_memory_stats()
for module_name, info in stats.items():
    print(f"Module: {module_name}")
    print(f"  Resident parameters: {info['resident_parameters']:,}")
    print(f"  Offloaded parameters: {info['offloaded_parameters']:,}")
    print(f"  Memory savings: {info['offloaded_parameters'] / info['total_parameters'] * 100:.1f}%")
```

### Disabling Offload

```python
# Disable offload and restore all parameters to GPU
model.disable_offload()
```

## Limitations

1. **FSDP Compatibility**: Designed specifically for FSDP scenarios
2. **Inference Only**: Optimized for inference workloads, not training
3. **Block-level Granularity**: Operates at the transformer block level
4. **Single GPU**: Currently designed for single-GPU scenarios

## Example Scripts

See `examples/stream_offload_example.py` for complete usage examples including both single-process and multiprocessing scenarios.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase the number of resident blocks
2. **Performance Degradation**: Ensure sufficient CPU-GPU memory bandwidth
3. **Hook Conflicts**: Avoid registering additional hooks on managed modules

### Logging

Enable INFO level logging to see offload initialization and memory transfer information:

```python
import logging
logging.basicConfig(level=logging.INFO)
``` 