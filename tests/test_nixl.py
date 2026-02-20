#!/usr/bin/env python
"""Test NIXL GDS Loader"""

import sys
import argparse
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import NIXLGDSLoader

def main():
    parser = argparse.ArgumentParser(description="Test NIXL GDS Loader")
    parser.add_argument(
        "--model",
        type=str,
        default="7b",
        choices=["0.6b", "7b"],
        help="Model size: 0.6b or 7b (default: 0.6b)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device (default: cuda:0)"
    )
    args = parser.parse_args()

    # Map model choice to path
    model_paths = {
        "0.6b": "./models/qwen3-0.6b",
        "7b": "./models/qwen2.5-7b"
    }
    model_path = model_paths[args.model]
    device = args.device

    print("\n" + "="*70)
    print("Testing NIXL GDS Loader")
    print("="*70)
    print(f"Model size: {args.model}")
    print(f"Model path: {model_path}")
    print(f"Device:     {device}")
    print("="*70)

    # Check model path
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("  Please run first: python scripts/download_model.py")
        return 1

    print(f"Model path: {model_path}")

    # Initialize Loader
    print("\nInitializing NIXL GDS Loader...")
    try:
        loader = NIXLGDSLoader(model_path, device=device)
        print(f"Loader initialized successfully")
        print(f"  NIXL status: {'available' if loader.nixl_available else 'unavailable (will use fallback)'}")
    except Exception as e:
        print(f"Loader initialization failed: {e}")
        return 1

    # Load model
    print("\nLoading model...")
    try:
        model = loader.load_model(torch_dtype=torch.bfloat16)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print stats
    info = loader.get_model_info()
    load_time = loader.load_stats['model_load_time']
    param_size_mb = info['param_size_mb']
    bandwidth_gbs = (param_size_mb / 1024) / load_time if load_time > 0 else 0

    print(f"\n{'='*70}")
    print("Performance Stats")
    print(f"{'='*70}")
    print(f"  Load time:   {load_time:.4f}s")
    print(f"  Param size:  {param_size_mb:.2f} MB")
    print(f"  Bandwidth:   {bandwidth_gbs:.3f} GB/s")
    print(f"  GPU memory:  {loader.load_stats['gpu_memory_mb']:.2f} MB")
    print(f"  CPU memory:  {loader.load_stats['cpu_memory_mb']:.2f} MB")
    print(f"  NIXL status: {'enabled' if loader.nixl_available else 'fallback mode'}")

    print(f"\nModel Info:")
    print(f"  Parameters:  {info['num_parameters']:,}")
    print(f"{'='*70}")

    # Test inference
    print("\nRunning inference test...")
    try:
        output = loader.test_inference(
            prompt="Hello, how are you?",
            max_new_tokens=20
        )
        print("Inference successful")
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
