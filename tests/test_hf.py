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
    print("测试 HuggingFace Loader")
    print("="*70)
    print(f"模型大小: {args.model}")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    print("="*70)

    # 检查模型路径
    if not Path(model_path).exists():
        print(f"\n✗ 模型不存在: {model_path}")
        return 1

    # 初始化
    loader = HFLoader(model_path, device=device)

    # 加载模型
    model = loader.load_model(torch_dtype=torch.bfloat16)

    # 输出统计
    print(f"\n{'='*70}")
    print("【记录这些数据】")
    print(f"{'='*70}")
    print(f"✓ 加载时间: {loader.load_stats['model_load_time']:.4f} 秒")
    print(f"✓ GPU 内存: {loader.load_stats['gpu_memory_mb']:.2f} MB")
    print(f"✓ CPU 内存: {loader.load_stats['cpu_memory_mb']:.2f} MB")
    print(f"{'='*70}\n")

    # 测试推理
    loader.test_inference("Hello, what is the capital of China?", max_new_tokens=20)

    return 0


if __name__ == "__main__":
    sys.exit(main())
