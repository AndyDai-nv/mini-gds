#!/usr/bin/env python
"""Test HuggingFace Loader"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.loaders import HFLoader


def main():
    parser = argparse.ArgumentParser(description="Test HuggingFace Loader")
    parser.add_argument(
        "--model",
        type=str,
        default="0.6b",
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
    print("Testing HuggingFace Loader")
    print("="*70)
    print(f"Model size: {args.model}")
    print(f"Model path: {model_path}")
    print(f"Device:     {device}")
    print("="*70)

    # Check model path
    if not Path(model_path).exists():
        print(f"\nModel not found: {model_path}")
        return 1

    # Initialize
    loader = HFLoader(model_path, device=device)

    # Load model
    model = loader.load_model(torch_dtype=torch.bfloat16)

    # Print stats
    print(f"\n{'='*70}")
    print("Performance Stats")
    print(f"{'='*70}")
    print(f"  Load time:   {loader.load_stats['model_load_time']:.4f}s")
    print(f"  GPU memory:  {loader.load_stats['gpu_memory_mb']:.2f} MB")
    print(f"  CPU memory:  {loader.load_stats['cpu_memory_mb']:.2f} MB")
    print(f"{'='*70}\n")

    # Test inference
    loader.test_inference("Hello, what is the capital of China?", max_new_tokens=20)

    return 0


if __name__ == "__main__":
    sys.exit(main())
