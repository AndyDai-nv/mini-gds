"""
NIXL GDS model loader using NVIDIA Inference Xfer Library.

NIXL provides a higher-level abstraction over cuFile/GDS with:
- Simplified API for file-to-GPU transfers
- Automatic backend selection (GDS, UCX, POSIX)
- Asynchronous transfer support
- Memory abstraction layer

Reference: https://github.com/ai-dynamo/nixl
"""

import os
import json
import torch
import struct
from pathlib import Path
from typing import Dict, Any, Optional, List

from transformers import AutoConfig, AutoModelForCausalLM
from ..utils.timer import Timer, get_gpu_memory_info, get_cpu_memory_info


class NIXLGDSLoader:
    """
    Model loader using NVIDIA Inference Xfer Library (NIXL) with GDS backend.

    NIXL provides a higher-level API over cuFile, with automatic backend
    selection and optimized file-to-GPU transfers.
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        """Initialize NIXL GDS loader.

        Args:
            model_path: Path to model directory containing safetensors files
            device: Target CUDA device
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.config = None
        self.load_stats: Dict[str, Any] = {}
        self.nixl_available = self._check_nixl_available()
        self.nixl_agent = None

    def _check_nixl_available(self) -> bool:
        """Check if NIXL is available (lightweight check, no agent creation)."""
        try:
            import nixl

            print("✓ NIXL library found")

            # Check if GDS plugin exists (no agent creation)
            try:
                # Just check if we can import, don't create agent yet
                # Agent creation is expensive (~1-2s), defer to actual load time
                print("✓ NIXL import successful")
                return True

            except Exception as e:
                print(f"⚠ NIXL check failed: {e}")
                return False

        except ImportError:
            print("⚠ NIXL not installed")
            print("  Install with: pip install nixl[cu12]")
            return False

    def _initialize_nixl_agent(self):
        """Initialize NIXL agent for transfers."""
        if not self.nixl_available:
            return None

        try:
            import nixl

            # Create unique agent name
            agent_name = f"model_loader_{os.getpid()}"
            self.nixl_agent = nixl.nixl_agent(agent_name)

            # Create GDS backend
            self.nixl_agent.create_backend("GDS")

            print(f"✓ NIXL agent created: {agent_name}")
            print(f"✓ GDS backend created")
            return self.nixl_agent

        except Exception as e:
            print(f"✗ Failed to create NIXL agent: {e}")
            return None

    def _cleanup_nixl_agent(self):
        """Clean up NIXL agent."""
        if self.nixl_agent is not None:
            try:
                del self.nixl_agent
                self.nixl_agent = None
            except Exception as e:
                print(f"Warning: Failed to clean up NIXL agent: {e}")

    def load_model(
        self,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> AutoModelForCausalLM:
        """Load model using NIXL GDS.

        Args:
            torch_dtype: Data type for model weights

        Returns:
            Loaded model
        """
        print(f"\n{'='*60}")
        print("NIXL GDS Loader")
        print(f"{'='*60}")
        print(f"Model Path: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Dtype: {torch_dtype}")
        print(f"NIXL Available: {self.nixl_available}")
        print(f"{'='*60}\n")

        # Record initial memory
        cpu_mem_start = get_cpu_memory_info()
        gpu_mem_start = get_gpu_memory_info(self.device)

        # Load config
        with Timer("Load Config") as t:
            self.config = AutoConfig.from_pretrained(str(self.model_path))
        print(f"✓ {t}")

        # Load model using optimized direct GPU loading
        with Timer("Load Model (Direct GPU)") as t:
            load_time = self._load_model_direct_gpu(torch_dtype)
        print(f"✓ {t}")

        self.load_stats["model_load_time"] = load_time

        # Record final memory
        cpu_mem_end = get_cpu_memory_info()
        gpu_mem_end = get_gpu_memory_info(self.device)

        # Compute memory deltas
        self.load_stats.update(
            {
                "cpu_memory_mb": cpu_mem_end["rss_mb"] - cpu_mem_start["rss_mb"],
                "gpu_memory_mb": gpu_mem_end["allocated_mb"] - gpu_mem_start["allocated_mb"],
                "total_time": load_time,
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

    def _load_model_direct_gpu(self, torch_dtype: torch.dtype) -> float:
        """Load model directly to GPU using NIXL GDS transfers.

        This method:
        1. Creates model skeleton on meta device (no memory allocation)
        2. Uses NIXL to transfer weights directly from NVMe to GPU (true GDS)
        3. Falls back to safetensors if NIXL is not available
        4. Properly handles non-checkpoint buffers (e.g., rotary_emb.inv_freq)
        """
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        import time

        start_time = time.perf_counter()

        # Find safetensors files
        safetensors_files = sorted(self.model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")

        print(f"  Found {len(safetensors_files)} safetensors file(s)")

        # Step 1: Create model skeleton on meta device (fast, no memory)
        print("  Creating model skeleton (meta device)...")
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                self.config,
                torch_dtype=torch_dtype,
            )

        # Step 2: Load weights - use NIXL GDS if available, else fallback
        if self.nixl_available:
            print("  Loading weights via NIXL GDS (true GPUDirect Storage)...")
            self._initialize_nixl_agent()
            try:
                self._load_weights_with_nixl_gds(safetensors_files, torch_dtype)
            except Exception as e:
                print(f"  ⚠ NIXL GDS failed: {e}, falling back to safetensors")
                import traceback
                traceback.print_exc()
                self._load_weights_fallback_method(safetensors_files, torch_dtype)
            finally:
                self._cleanup_nixl_agent()
        else:
            print("  Loading weights via safetensors (NIXL not available)...")
            self._load_weights_fallback_method(safetensors_files, torch_dtype)

        # Step 3: Handle non-checkpoint buffers (e.g., rotary embeddings)
        #print("  Initializing remaining buffers on GPU...")
        #self._materialize_remaining_buffers(torch_dtype)

        # Verify model is on correct device
        #self.model = self.model.to(self.device)

        elapsed = time.perf_counter() - start_time
        print(f"  Total load time: {elapsed:.4f}s")
        return elapsed

    def _load_weights_with_nixl_gds(self, safetensors_files: list, torch_dtype: torch.dtype):
        """Load weights using NIXL GDS - true file-to-GPU DMA transfers.

        Optimized to reuse file descriptor across all tensors.
        """
        from accelerate.utils import set_module_tensor_to_device
        import os
        import gc

        for sf_file in safetensors_files:
            print(f"    Loading: {sf_file.name} (via NIXL GDS)")

            # Parse safetensors file header once
            header_len, tensors_metadata = self._parse_safetensors_header(sf_file)

            # Open file once for all tensors
            fd = os.open(str(sf_file), os.O_RDONLY)

            try:
                # Load each tensor using the same file descriptor
                for tensor_name, metadata in tensors_metadata.items():
                    if tensor_name == "__metadata__":
                        continue

                    # Load tensor using pre-opened fd
                    tensor = self._load_tensor_with_fd(
                        fd,
                        header_len,
                        metadata,
                        torch_dtype
                    )

                    # Set tensor to model
                    set_module_tensor_to_device(
                        self.model, tensor_name, self.device, value=tensor, non_blocking=True
                    )

                    # Clear reference to help GC
                    del tensor

            finally:
                # Close file once after all tensors loaded
                os.close(fd)

                # Clear metadata to free memory
                del tensors_metadata
                gc.collect()

    def _parse_safetensors_header(self, file_path: Path) -> tuple:
        """Parse safetensors file header to get tensor metadata.

        Safetensors format:
        - Bytes 0-7: little-endian uint64 (header length)
        - Bytes 8 to 8+header_len: JSON header
        - Remaining bytes: tensor data

        Returns:
            (header_len, metadata_dict)
            - header_len: int, length of header in bytes
            - metadata_dict: dict mapping tensor names to {dtype, shape, data_offsets}
        """
        import json

        with open(file_path, 'rb') as f:
            # Read header length (first 8 bytes, little-endian uint64)
            header_len_bytes = f.read(8)
            header_len = int.from_bytes(header_len_bytes, byteorder='little')

            # Read JSON header
            header_json = f.read(header_len).decode('utf-8')
            metadata = json.loads(header_json)

        return header_len, metadata

    def _load_tensor_with_fd(
        self,
        fd: int,
        header_len: int,
        metadata: dict,
        torch_dtype: torch.dtype
    ) -> torch.Tensor:
        """Load a single tensor using NIXL file-to-GPU transfer with existing fd.

        Args:
            fd: Open file descriptor (from os.open)
            header_len: Length of safetensors header in bytes
            metadata: Tensor metadata from safetensors header
                     {"dtype": "F32", "shape": [768, 768], "data_offsets": [0, 2359296]}
            torch_dtype: Target PyTorch dtype

        Returns:
            Tensor loaded directly to GPU via NIXL GDS (DMA transfer)
        """
        if self.nixl_agent is None:
            raise RuntimeError("NIXL agent not initialized")

        import time

        # Parse metadata
        shape = tuple(metadata["shape"])
        data_offsets = metadata["data_offsets"]
        begin_offset = data_offsets[0]
        end_offset = data_offsets[1]
        tensor_size_bytes = end_offset - begin_offset

        # Calculate absolute file offset (8 bytes + header_len + data_offset)
        absolute_file_offset = 8 + header_len + begin_offset

        # Allocate GPU tensor
        tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)

        # Use provided file descriptor (no need to open/close)
        try:
            # Register file: (offset, size, fd, label)
            file_list = [(absolute_file_offset, tensor_size_bytes, fd, f"tensor_{absolute_file_offset}")]
            file_descs = self.nixl_agent.register_memory(file_list, "FILE")

            if file_descs is None:
                raise RuntimeError("Failed to register file with NIXL")

            # Get file transfer descriptors
            file_xfer_descs = file_descs.trim()

            # Register GPU tensor
            gpu_descs = self.nixl_agent.register_memory(tensor, "VRAM")

            if gpu_descs is None:
                raise RuntimeError("Failed to register GPU memory with NIXL")

            # Get GPU transfer descriptors
            gpu_xfer_descs = self.nixl_agent.get_xfer_descs(tensor, "VRAM")

            # Initialize transfer: READ from file to GPU
            # For local transfers, remote_agent is self
            xfer_handle = self.nixl_agent.initialize_xfer(
                "READ",                    # Read from file
                gpu_xfer_descs,           # Destination (GPU)
                file_xfer_descs,          # Source (file)
                self.nixl_agent.name      # Local transfer
            )

            if not xfer_handle:
                raise RuntimeError("Failed to initialize NIXL transfer")

            # Execute transfer (async DMA: file → GPU)
            state = self.nixl_agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("Failed to start NIXL transfer")

            # Wait for completion
            max_wait_seconds = 30
            wait_start = time.time()

            while True:
                state = self.nixl_agent.check_xfer_state(xfer_handle)

                if state == "DONE":
                    break
                elif state == "ERR":
                    raise RuntimeError(f"NIXL transfer failed for tensor at offset {absolute_file_offset}")

                if time.time() - wait_start > max_wait_seconds:
                    raise TimeoutError(f"NIXL transfer timeout after {max_wait_seconds}s")

                time.sleep(0.001)  # 1ms polling

            # Release resources
            self.nixl_agent.release_xfer_handle(xfer_handle)
            self.nixl_agent.deregister_memory(file_descs)
            self.nixl_agent.deregister_memory(gpu_descs)

        except Exception as e:
            # Let caller handle the error (don't close fd here)
            raise

        return tensor

    def _load_weights_fallback_method(self, safetensors_files: list, torch_dtype: torch.dtype):
        """Fallback: load weights using safetensors (no NIXL)."""
        from safetensors.torch import load_file
        from accelerate.utils import set_module_tensor_to_device

        for sf_file in safetensors_files:
            print(f"    Loading: {sf_file.name} (fallback)")
            file_state_dict = load_file(str(sf_file), device=self.device)

            for key, tensor in file_state_dict.items():
                tensor = tensor.to(dtype=torch_dtype)
                set_module_tensor_to_device(
                    self.model, key, self.device, value=tensor
                )
    
    def _materialize_remaining_buffers(self, torch_dtype: torch.dtype):
        """Initialize buffers that are not in the checkpoint.
        
        Some models have buffers (like rotary embedding inv_freq) that are
        computed during __init__ and not saved. We need to recreate these.
        """
        device = torch.device(self.device)
        
        for name, module in self.model.named_modules():
            # Handle rotary embeddings (common in Llama/Qwen models)
            if hasattr(module, 'inv_freq') and module.inv_freq.device.type == 'meta':
                # Recreate inv_freq buffer
                dim = module.dim if hasattr(module, 'dim') else module.inv_freq.shape[0] * 2
                base = module.base if hasattr(module, 'base') else 10000.0
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                module.register_buffer('inv_freq', inv_freq.to(device=device, dtype=torch_dtype), persistent=False)
            
            # Handle any other meta tensors in parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param.device.type == 'meta':
                    # Initialize with zeros (will be overwritten if in checkpoint)
                    new_param = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=torch_dtype, device=device)
                    )
                    setattr(module, param_name, new_param)
            
            # Handle any other meta tensors in buffers
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf is not None and buf.device.type == 'meta':
                    # Try to infer how to initialize
                    new_buf = torch.zeros(buf.shape, dtype=torch_dtype, device=device)
                    module.register_buffer(buf_name, new_buf)


    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"error": "Model not loaded"}

        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2

        return {
            "model_path": str(self.model_path),
            "num_parameters": param_count,
            "param_size_mb": param_size_mb,
            "num_layers": self.config.num_hidden_layers if self.config else None,
            "hidden_size": self.config.hidden_size if self.config else None,
            "vocab_size": self.config.vocab_size if self.config else None,
            "nixl_available": self.nixl_available,
        }

    @torch.no_grad()
    def test_inference(
        self, prompt: str = "Hello, how are you?", max_new_tokens: int = 20
    ) -> str:
        """Test inference with a sample prompt.

        Args:
            prompt: Input text
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model must be loaded first")

        from transformers import AutoTokenizer

        print(f"\n{'='*60}")
        print("Test Inference")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with Timer("Generate") as t:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"✓ {t}")
        print(f"Output: {generated_text}")
        print(f"{'='*60}\n")

        return generated_text
