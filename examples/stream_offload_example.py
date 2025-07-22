#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Example script demonstrating the streaming offload functionality in WanVace model.
This feature helps reduce GPU memory usage by keeping only a subset of model blocks
resident on GPU and streaming others from CPU as needed.
"""

import logging
import torch
from wan.vace import WanVace, WanVaceMP
from wan.configs.wan_t2v_14B import config

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Example configuration
    checkpoint_dir = "path/to/your/checkpoint"
    device_id = 0
    
    print("=== WanVace Streaming Offload Example ===\n")
    
    # 1. Initialize WanVace with streaming offload enabled
    print("1. Initializing WanVace with streaming offload...")
    model = WanVace(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=0,
        enable_offload=True,           # Enable streaming offload
        blocks_resident_count=2,       # Keep 2 main blocks resident on GPU
        vace_blocks_resident_count=2   # Keep 2 vace blocks resident on GPU
    )
    
    # 2. Check offload status
    print(f"2. Offload enabled: {model.is_offload_enabled()}")
    
    # 3. Get memory statistics
    if model.is_offload_enabled():
        stats = model.get_offload_memory_stats()
        print("3. Memory statistics:")
        for module_name, module_stats in stats.items():
            print(f"   {module_name}:")
            print(f"     - Total blocks: {module_stats['total_blocks']}")
            print(f"     - Resident blocks: {module_stats['resident_blocks']}")
            print(f"     - Offloaded blocks: {module_stats['offloaded_blocks']}")
            print(f"     - Resident parameters: {module_stats['resident_parameters']:,}")
            print(f"     - Offloaded parameters: {module_stats['offloaded_parameters']:,}")
            print(f"     - Total parameters: {module_stats['total_parameters']:,}")
    
    # 4. Example generation (you would provide your actual inputs here)
    print("\n4. Running example generation with streaming offload...")
    
    # Dummy inputs for demonstration
    input_prompt = "A beautiful sunset over the ocean"
    # Note: You need to provide actual input_frames, input_masks, input_ref_images
    # input_frames = [your_frames]
    # input_masks = [your_masks] 
    # input_ref_images = [your_ref_images]
    
    # Uncomment and modify when you have actual inputs:
    # video = model.generate(
    #     input_prompt=input_prompt,
    #     input_frames=input_frames,
    #     input_masks=input_masks,
    #     input_ref_images=input_ref_images,
    #     size=(1280, 720),
    #     frame_num=81,
    #     sampling_steps=50,
    #     guide_scale=5.0,
    #     seed=42
    # )
    
    print("Generation would run here with streaming offload active...")
    
    # 5. Disable offload if needed
    print("\n5. Disabling streaming offload...")
    model.disable_offload()
    print(f"   Offload enabled after disable: {model.is_offload_enabled()}")
    
    print("\n=== Example completed ===")

def multiprocessing_example():
    """
    Example showing how to use streaming offload with WanVaceMP (multiprocessing version).
    """
    print("\n=== WanVaceMP Streaming Offload Example ===\n")
    
    checkpoint_dir = "path/to/your/checkpoint"
    
    # Initialize WanVaceMP with streaming offload
    model_mp = WanVaceMP(
        config=config,
        checkpoint_dir=checkpoint_dir,
        use_usp=False,
        enable_offload=True,           # Enable streaming offload
        blocks_resident_count=2,       # Keep 2 main blocks resident on GPU
        vace_blocks_resident_count=2   # Keep 2 vace blocks resident on GPU
    )
    
    # Check configuration
    print(f"Offload configured: {model_mp.is_offload_enabled()}")
    offload_config = model_mp.get_offload_config()
    print(f"Offload config: {offload_config}")
    
    # Note: Each worker process will automatically initialize its own OffloadManager
    # with the specified configuration
    
    print("Each worker process will have its own streaming offload enabled")

if __name__ == "__main__":
    # Run the basic example
    main()
    
    # Uncomment to run multiprocessing example
    # multiprocessing_example() 