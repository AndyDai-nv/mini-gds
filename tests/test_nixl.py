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
    print("测试 NIXL GDS Loader")
    print("="*70)
    print(f"模型大小: {args.model}")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    print("="*70)

    # 检查模型路径
    if not Path(model_path).exists():
        print(f"✗ 模型不存在: {model_path}")
        print("  请先运行: python scripts/download_model.py")
        return 1
    
    print(f"✓ 模型路径: {model_path}")
    
    # 初始化 Loader
    print("\n初始化 NIXL GDS Loader...")
    try:
        loader = NIXLGDSLoader(model_path, device=device)
        print(f"✓ Loader 初始化成功")
        print(f"  NIXL 状态: {'可用' if loader.nixl_available else '不可用（将使用回退）'}")
    except Exception as e:
        print(f"✗ Loader 初始化失败: {e}")
        return 1
    
    # 加载模型
    print("\n开始加载模型...")
    try:
        model = loader.load_model(torch_dtype=torch.bfloat16)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 输出统计
    print(f"\n{'='*70}")
    print("【性能统计 - 记录这些数据】")
    print(f"{'='*70}")
    print(f"✓ 加载时间: {loader.load_stats['model_load_time']:.4f} 秒")
    print(f"✓ GPU 内存: {loader.load_stats['gpu_memory_mb']:.2f} MB")
    print(f"✓ CPU 内存: {loader.load_stats['cpu_memory_mb']:.2f} MB")
    print(f"✓ NIXL 状态: {'启用' if loader.nixl_available else '回退模式'}")
    
    # 获取模型信息
    info = loader.get_model_info()
    print(f"\n模型信息:")
    print(f"  参数量: {info['num_parameters']:,}")
    print(f"  参数大小: {info['param_size_mb']:.2f} MB")
    print(f"{'='*70}")
    
    # 测试推理
    print("\n测试推理...")
    try:
        output = loader.test_inference(
            prompt="Hello, how are you?",
            max_new_tokens=20
        )
        print("✓ 推理成功")
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("🎉 所有测试通过！")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
