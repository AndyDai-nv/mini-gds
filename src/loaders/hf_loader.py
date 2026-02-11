"""Traditional HuggingFace model loader."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ..utils.timer import Timer, get_gpu_memory_info, get_cpu_memory_info


class HFLoader:
    """HuggingFace model loader with profiling."""

    def __init__(self, model_name_or_path: str, device: str = "cuda:0"):
        """Initialize loader.

        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Target device for model
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_stats: Dict[str, Any] = {}

    def load_model(
        self,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_safetensors: bool = True,
        low_cpu_mem_usage: bool = False,
    ) -> AutoModelForCausalLM:
        """Load model using HuggingFace transformers.

        Args:
            torch_dtype: Data type for model weights
            use_safetensors: Whether to use safetensors format
            low_cpu_mem_usage: Enable low CPU memory mode

        Returns:
            Loaded model
        """
        print(f"\n{'='*60}")
        print("HuggingFace Loader")
        print(f"{'='*60}")
        print(f"Model: {self.model_name_or_path}")
        print(f"Device: {self.device}")
        print(f"Dtype: {torch_dtype}")
        print(f"SafeTensors: {use_safetensors}")
        print(f"{'='*60}\n")

        # Record initial memory
        cpu_mem_start = get_cpu_memory_info()
        gpu_mem_start = get_gpu_memory_info(self.device)

        # Load config
        with Timer("Load Config") as t:
            self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        print(f"✓ {t}")

        # Load model
        with Timer("Load Model") as t:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                torch_dtype=torch_dtype,
                device_map=self.device,
                use_safetensors=use_safetensors,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        self.load_stats["model_load_time"] = t.elapsed
        print(f"✓ {t}")

        # Load tokenizer
        with Timer("Load Tokenizer") as t:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        print(f"✓ {t}")

        # Record final memory
        cpu_mem_end = get_cpu_memory_info()
        gpu_mem_end = get_gpu_memory_info(self.device)

        # Compute memory deltas
        self.load_stats.update(
            {
                "cpu_memory_mb": cpu_mem_end["rss_mb"] - cpu_mem_start["rss_mb"],
                "gpu_memory_mb": gpu_mem_end["allocated_mb"] - gpu_mem_start["allocated_mb"],
                "total_time": t.elapsed,
            }
        )

        print(f"\n{'='*60}")
        print("Loading Statistics")
        print(f"{'='*60}")
        print(f"Total Time: {self.load_stats['model_load_time']:.4f}s")
        print(f"GPU Memory: {self.load_stats['gpu_memory_mb']:.2f} MB")
        print(f"CPU Memory: {self.load_stats['cpu_memory_mb']:.2f} MB")
        print(f"{'='*60}\n")

        return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"error": "Model not loaded"}

        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2

        return {
            "model_name": self.model_name_or_path,
            "num_parameters": param_count,
            "param_size_mb": param_size_mb,
            "num_layers": self.config.num_hidden_layers if self.config else None,
            "hidden_size": self.config.hidden_size if self.config else None,
            "vocab_size": self.config.vocab_size if self.config else None,
        }

    @torch.no_grad()
    def test_inference(self, prompt: str = "Hello, how are you?", max_new_tokens: int = 20) -> str:
        """Test inference with a sample prompt.

        Args:
            prompt: Input text
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        print(f"\n{'='*60}")
        print("Test Inference")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with Timer("Generate") as t:
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"✓ {t}")
        print(f"Output: {generated_text}")
        print(f"{'='*60}\n")

        return generated_text
