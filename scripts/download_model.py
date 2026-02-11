"""
Download Qwen3-0.6B model from HuggingFace Hub using snapshot_download.

This script efficiently downloads the model to a local directory for use with GDS loader.
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "./models/qwen3-0.6b",
    allow_patterns: list = None,
    ignore_patterns: list = None,
):
    """Download model from HuggingFace Hub using snapshot_download.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Local directory to save model
        allow_patterns: List of file patterns to download (e.g., ["*.safetensors", "*.json"])
        ignore_patterns: List of file patterns to ignore
    """
    output_path = Path(output_dir).absolute()

    print(f"\n{'='*60}")
    print("Downloading Model from HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")

    # Default: download safetensors, config, tokenizer files
    if allow_patterns is None:
        allow_patterns = [
            "*.safetensors",      # Model weights (SafeTensors format)
            "*.json",             # Config files
            "*.txt",              # Vocab files
            "*.model",            # Tokenizer model
            "tokenizer.model",    # SentencePiece model
            "*.tiktoken",         # Tiktoken files
        ]

    # Ignore unnecessary files
    if ignore_patterns is None:
        ignore_patterns = [
            "*.bin",              # PyTorch format (we prefer safetensors)
            "*.h5",               # TensorFlow format
            "*.msgpack",          # Flax format
            "*.ot",               # ONNX format
            "pytorch_model.bin*", # Old PyTorch format
        ]

    print(f"File patterns:")
    print(f"  Allow: {', '.join(allow_patterns)}")
    print(f"  Ignore: {', '.join(ignore_patterns)}")
    print(f"{'='*60}\n")

    print("Downloading files...")
    print("(This may take a few minutes depending on model size and network speed)\n")

    # Download using snapshot_download
    # This API handles:
    # - Parallel downloads
    # - Resume capability
    # - Caching
    # - Efficient file filtering
    local_path = snapshot_download(
        repo_id=model_name,
        local_dir=output_path,
        local_dir_use_symlinks=False,  # Copy files instead of symlinks
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=True,           # Resume if interrupted
    )

    print(f"\n{'='*60}")
    print("Download Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {local_path}")

    # List downloaded files
    print("\nDownloaded files:")
    for file in sorted(output_path.rglob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"  {file.name:<40} {size_mb:>8.2f} MB")

    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"{'='*60}")
    print(f"1. Verify installation:")
    print(f"   python scripts/quick_test.py")
    print(f"\n2. Run benchmark:")
    print(f"   python benchmarks/benchmark_loader.py --model {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download model from HuggingFace Hub using snapshot_download"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/qwen3-0.6b",
        help="Output directory (default: ./models/qwen3-0.6b)",
    )
    parser.add_argument(
        "--include-pytorch",
        action="store_true",
        help="Also download PyTorch .bin files (default: safetensors only)",
    )
    parser.add_argument(
        "--allow-patterns",
        type=str,
        nargs="+",
        help="Custom file patterns to download (e.g., '*.safetensors' '*.json')",
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        nargs="+",
        help="File patterns to ignore",
    )

    args = parser.parse_args()

    # Adjust ignore patterns if user wants PyTorch files
    ignore_patterns = args.ignore_patterns
    if args.include_pytorch and ignore_patterns is None:
        ignore_patterns = [
            "*.h5",               # TensorFlow format
            "*.msgpack",          # Flax format
            "*.ot",               # ONNX format
        ]

    download_model(
        model_name=args.model,
        output_dir=args.output,
        allow_patterns=args.allow_patterns,
        ignore_patterns=ignore_patterns,
    )


if __name__ == "__main__":
    main()
