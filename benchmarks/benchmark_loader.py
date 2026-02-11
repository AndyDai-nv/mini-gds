"""
Benchmark script to compare HuggingFace loader vs GDS loader.

This script measures:
1. Model loading time
2. Memory usage (CPU and GPU)
3. First inference latency
4. I/O characteristics
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import HFLoader, GDSLoader
from src.utils.timer import Timer, get_gpu_memory_info, get_cpu_memory_info


def benchmark_hf_loader(
    model_name: str, device: str = "cuda:0", torch_dtype: torch.dtype = torch.bfloat16
) -> Dict[str, Any]:
    """Benchmark HuggingFace loader.

    Args:
        model_name: HuggingFace model name or local path
        device: CUDA device
        torch_dtype: Model dtype

    Returns:
        Benchmark results dictionary
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: HuggingFace Loader")
    print("=" * 80)

    results = {"loader": "huggingface", "model": model_name, "device": device}

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Initialize loader
    loader = HFLoader(model_name, device=device)

    # Load model
    try:
        model = loader.load_model(torch_dtype=torch_dtype, use_safetensors=True)
        results.update(loader.load_stats)
        results.update(loader.get_model_info())

        # Test inference
        try:
            loader.test_inference(
                prompt="The future of AI is", max_new_tokens=20
            )
        except Exception as e:
            print(f"⚠ Inference test failed: {e}")

        results["success"] = True

    except Exception as e:
        print(f"✗ Loading failed: {e}")
        results["success"] = False
        results["error"] = str(e)

    return results


def benchmark_gds_loader(
    model_path: str, device: str = "cuda:0", torch_dtype: torch.dtype = torch.bfloat16
) -> Dict[str, Any]:
    """Benchmark GDS loader.

    Args:
        model_path: Local path to model directory
        device: CUDA device
        torch_dtype: Model dtype

    Returns:
        Benchmark results dictionary
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: GDS Loader")
    print("=" * 80)

    results = {"loader": "gds", "model": model_path, "device": device}

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Initialize loader
    loader = GDSLoader(model_path, device=device)

    # Load model
    try:
        model = loader.load_model(torch_dtype=torch_dtype)
        results.update(loader.load_stats)
        results.update(loader.get_model_info())

        # Test inference
        try:
            # Create sample input
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            loader.test_inference(input_ids, max_new_tokens=20)
        except Exception as e:
            print(f"⚠ Inference test failed: {e}")

        results["success"] = True

    except Exception as e:
        print(f"✗ Loading failed: {e}")
        results["success"] = False
        results["error"] = str(e)

    return results


def print_comparison(hf_results: Dict[str, Any], gds_results: Dict[str, Any]):
    """Print comparison table.

    Args:
        hf_results: HuggingFace loader results
        gds_results: GDS loader results
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Loading time comparison
    if hf_results.get("success") and gds_results.get("success"):
        hf_time = hf_results.get("model_load_time", 0)
        gds_time = gds_results.get("model_load_time", 0)
        speedup = hf_time / gds_time if gds_time > 0 else 0

        print(f"\n{'Metric':<30} {'HuggingFace':<20} {'GDS':<20} {'Speedup':<15}")
        print("-" * 85)

        print(
            f"{'Loading Time (s)':<30} {hf_time:<20.4f} {gds_time:<20.4f} {speedup:<15.2f}x"
        )

        hf_gpu = hf_results.get("gpu_memory_mb", 0)
        gds_gpu = gds_results.get("gpu_memory_mb", 0)
        print(
            f"{'GPU Memory (MB)':<30} {hf_gpu:<20.2f} {gds_gpu:<20.2f} {'-':<15}"
        )

        hf_cpu = hf_results.get("cpu_memory_mb", 0)
        gds_cpu = gds_results.get("cpu_memory_mb", 0)
        cpu_reduction = (
            (hf_cpu - gds_cpu) / hf_cpu * 100 if hf_cpu > 0 else 0
        )
        print(
            f"{'CPU Memory (MB)':<30} {hf_cpu:<20.2f} {gds_cpu:<20.2f} {f'-{cpu_reduction:.1f}%':<15}"
        )

        hf_params = hf_results.get("num_parameters", 0)
        gds_params = gds_results.get("num_parameters", 0)
        print(
            f"{'Parameters (M)':<30} {hf_params/1e6:<20.2f} {gds_params/1e6:<20.2f} {'-':<15}"
        )

        print("\n" + "=" * 80)
        print(f"SUMMARY: GDS is {speedup:.2f}x faster than HuggingFace loader")
        print(f"         CPU memory reduced by {cpu_reduction:.1f}%")
        print("=" * 80 + "\n")

    else:
        print("\n⚠ Could not generate comparison - one or both loaders failed")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model loaders")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HuggingFace loader benchmark",
    )
    parser.add_argument(
        "--skip-gds",
        action="store_true",
        help="Skip GDS loader benchmark",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA not available. Exiting.")
        return

    print("\n" + "=" * 80)
    print("MODEL LOADER BENCHMARK")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print("=" * 80)

    results = {}

    # Benchmark HuggingFace loader
    if not args.skip_hf:
        hf_results = benchmark_hf_loader(args.model, args.device, torch_dtype)
        results["huggingface"] = hf_results
    else:
        print("\n⊘ Skipping HuggingFace loader benchmark")
        hf_results = None

    # Benchmark GDS loader
    if not args.skip_gds:
        # For GDS, we need a local path
        model_path = Path(args.model)
        if not model_path.exists():
            print(
                f"\n⚠ Model path {args.model} does not exist locally."
            )
            print("  GDS loader requires a local model directory.")
            print("  Run: python scripts/download_model.py first")
            gds_results = None
        else:
            gds_results = benchmark_gds_loader(str(model_path), args.device, torch_dtype)
            results["gds"] = gds_results
    else:
        print("\n⊘ Skipping GDS loader benchmark")
        gds_results = None

    # Print comparison
    if hf_results and gds_results:
        print_comparison(hf_results, gds_results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
