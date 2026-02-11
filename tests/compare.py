#!/usr/bin/env python
"""Compare HF vs NIXL GDS loaders."""

import sys
import argparse
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import HFLoader, NIXLGDSLoader


def test_loader(loader_type: str, model_path: str, device: str):
    """Test a specific loader."""
    print("\n" + "="*70)
    print(f"Testing {loader_type} Loader")
    print("="*70)

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if loader_type == "HF":
        loader = HFLoader(model_path, device=device)
        model = loader.load_model(
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
    else:  # NIXL
        loader = NIXLGDSLoader(model_path, device=device)
        if not loader.nixl_available:
            print("NIXL not available")
            return None
        model = loader.load_model(torch_dtype=torch.bfloat16)

    stats = {
        'time': loader.load_stats['model_load_time'],
        'gpu_mem': loader.load_stats['gpu_memory_mb'],
        'cpu_mem': loader.load_stats['cpu_memory_mb']
    }

    print("\n" + "="*70)
    print(f"{loader_type} Loader Results")
    print("="*70)
    print(f"  Load time:   {stats['time']:.4f}s")
    print(f"  GPU memory:  {stats['gpu_mem']:.2f} MB")
    print(f"  CPU memory:  {stats['cpu_mem']:.2f} MB")
    print("="*70)

    # Quick inference test
    if loader_type == "HF":
        print("\nRunning inference test...")
        loader.test_inference("Hello, what is AI?", max_new_tokens=20)

    del model
    torch.cuda.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare HF vs NIXL GDS Loaders")
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
    parser.add_argument(
        "--loader",
        type=str,
        default="both",
        choices=["hf", "nixl", "both"],
        help="Which loader to test (default: both)"
    )
    args = parser.parse_args()

    # Map model choice to path
    model_paths = {
        "0.6b": "./models/qwen3-0.6b",
        "7b": "./models/qwen2.5-7b"
    }
    model_path = model_paths[args.model]

    # Get expected tensor info
    tensor_info = {
        "0.6b": {"count": 311, "size": "~1.4GB", "avg": "~5MB"},
        "7b": {"count": 339, "size": "~14GB", "avg": "~41MB"}
    }
    info = tensor_info[args.model]

    print("\n" + "="*70)
    print("Loader Performance Comparison")
    print("="*70)
    print(f"Model:   {args.model} ({model_path})")
    print(f"Device:  {args.device}")
    print(f"Tensors: {info['count']} tensors, {info['size']} total, {info['avg']} avg")
    print("="*70)

    if not torch.cuda.is_available():
        print("\nCUDA not available")
        return 1

    hf_stats = None
    nixl_stats = None

    # Test HF
    if args.loader in ["hf", "both"]:
        print("\nNote: Clear page cache before testing for cold start")
        print("  echo 3 | sudo tee /proc/sys/vm/drop_caches")
        input("\nPress Enter after clearing cache to test HF...")
        hf_stats = test_loader("HF", model_path, args.device)
        if args.loader == "both":
            print("\nWaiting 5 seconds...")
            time.sleep(5)

    # Test NIXL
    if args.loader in ["nixl", "both"]:
        if args.loader == "both":
            print("\nNote: Clear cache again for fair comparison")
            print("  echo 3 | sudo tee /proc/sys/vm/drop_caches")
            input("\nPress Enter after clearing cache to test NIXL...")
        nixl_stats = test_loader("NIXL", model_path, args.device)

    # Comparison
    if hf_stats and nixl_stats:
        print("\n" + "="*70)
        print("Final Comparison")
        print("="*70)

        print(f"\nLoad Time:")
        print(f"  HF:   {hf_stats['time']:.4f}s")
        print(f"  NIXL: {nixl_stats['time']:.4f}s")
        ratio = nixl_stats['time'] / hf_stats['time']
        if ratio < 1:
            print(f"  NIXL is {1/ratio:.2f}x faster")
        else:
            print(f"  NIXL is {ratio:.2f}x slower")

        print(f"\nCPU Memory:")
        print(f"  HF:   {hf_stats['cpu_mem']:.2f} MB")
        print(f"  NIXL: {nixl_stats['cpu_mem']:.2f} MB")
        savings = hf_stats['cpu_mem'] - nixl_stats['cpu_mem']
        savings_pct = (savings / hf_stats['cpu_mem']) * 100
        print(f"  Savings: {savings:.2f} MB ({savings_pct:.1f}%)")

        print(f"\nGPU Memory:")
        print(f"  HF:   {hf_stats['gpu_mem']:.2f} MB")
        print(f"  NIXL: {nixl_stats['gpu_mem']:.2f} MB")

        # Result for 7B
        if args.model == "7b":
            print("\n" + "="*70)
            print(f"Result: NIXL {nixl_stats['time']:.2f}s vs HF {hf_stats['time']:.2f}s ", end="")
            if ratio < 1:
                print(f"(NIXL {1/ratio:.2f}x faster)")
            else:
                print(f"(NIXL {ratio:.2f}x slower)")

        print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
