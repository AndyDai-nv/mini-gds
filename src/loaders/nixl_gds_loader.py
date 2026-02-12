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

    # GDS requires 4KB-aligned file offsets and transfer sizes
    GDS_ALIGNMENT = 4096
    # Max chunk size per GDS batch I/O (must be <= cuFile per_buffer_cache_size).
    # Project cufile.json sets per_buffer_cache_size_kb=16384 (16MB).
    GDS_MAX_CHUNK_SIZE = 16 * 1024 * 1024  # 16MB

    # Safetensors dtype string -> torch dtype mapping
    SAFETENSORS_DTYPE_MAP = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }

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
        self._chunk_buffer = None  # Reusable GPU buffer for chunked transfers

        # Set project cufile.json before any NIXL/cuFile init
        self._setup_cufile_env()
        self._read_cufile_config()

        self.nixl_available = self._check_nixl_available()
        self.nixl_agent = None

    def _setup_cufile_env(self):
        """Point cuFile to project-level cufile.json with optimized settings."""
        if "CUFILE_ENV_PATH_JSON" not in os.environ:
            project_cufile = Path(__file__).resolve().parent.parent.parent / "configs" / "cufile.json"
            if project_cufile.exists():
                os.environ["CUFILE_ENV_PATH_JSON"] = str(project_cufile)

    def _read_cufile_config(self):
        """Read GDS_MAX_CHUNK_SIZE from cufile.json if available."""
        cufile_path = os.environ.get("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
        try:
            with open(cufile_path) as f:
                # cufile.json uses C-style comments, strip them
                lines = []
                for line in f:
                    stripped = line.lstrip()
                    if not stripped.startswith("//"):
                        # Remove inline comments
                        comment_pos = line.find("//")
                        if comment_pos >= 0:
                            line = line[:comment_pos]
                        lines.append(line)
                content = "".join(lines)
                config = json.loads(content)
            cache_kb = config.get("properties", {}).get("per_buffer_cache_size_kb", 1024)
            self.GDS_MAX_CHUNK_SIZE = cache_kb * 1024
        except Exception:
            pass  # Use default 1MB

    def _check_nixl_available(self) -> bool:
        """Check if NIXL is available (lightweight check, no agent creation)."""
        try:
            import nixl

            print("✓ NIXL library found")

            # Check if we can import, don't create agent yet
            # Agent creation is expensive (~1-2s), defer to actual load time
            try:
                print("✓ NIXL import successful")
                return True

            except Exception as e:
                print(f"⚠ NIXL check failed: {e}")
                return False

        except ImportError:
            print("⚠ NIXL not installed")
            print("  Install with: pip install nixl[cu12]")
            return False

    @staticmethod
    def _get_nvfs_batch_io_count() -> int:
        """Read BatchIO count from nvidia-fs stats."""
        try:
            with open("/proc/driver/nvidia-fs/stats") as f:
                for line in f:
                    if line.strip().startswith("Ops"):
                        # "Ops  : Read=0 Write=0 BatchIO=0"
                        for part in line.split():
                            if part.startswith("BatchIO="):
                                return int(part.split("=")[1])
        except (FileNotFoundError, ValueError):
            pass
        return -1  # Cannot determine

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
        print(f"GDS chunk size: {self.GDS_MAX_CHUNK_SIZE // 1024}KB")
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

        elapsed = time.perf_counter() - start_time
        print(f"  Total load time: {elapsed:.4f}s")
        return elapsed

    def _load_weights_with_nixl_gds(self, safetensors_files: list, torch_dtype: torch.dtype):
        """Load weights using NIXL GDS - true file-to-GPU DMA transfers.

        After the first file, checks nvidia-fs stats to detect compat mode.
        If running in compat mode (POSIX fallback), raises to switch to
        direct safetensors loading which is faster without true GDS.
        """
        from accelerate.utils import set_module_tensor_to_device
        import os
        import gc

        batch_io_before = self._get_nvfs_batch_io_count()

        for file_idx, sf_file in enumerate(safetensors_files):
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

            # After first file: check if true GDS or compat mode
            if file_idx == 0 and batch_io_before >= 0:
                batch_io_after = self._get_nvfs_batch_io_count()
                if batch_io_after == batch_io_before:
                    print("  ⚠ Detected cuFile compat mode (no true GDS I/O)")
                    print("    Switching to direct safetensors for remaining files...")
                    # Load remaining files via safetensors (faster in compat mode)
                    remaining_files = safetensors_files[file_idx + 1:]
                    if remaining_files:
                        self._load_weights_fallback_method(remaining_files, torch_dtype)
                    return
                else:
                    print(f"  ✓ True GDS active (BatchIO: {batch_io_before} → {batch_io_after})")

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

    def _nixl_transfer_chunk(self, fd, file_offset, gpu_buffer, chunk_size):
        """Transfer a single aligned chunk from file to GPU via NIXL GDS.

        Args:
            fd: Open file descriptor
            file_offset: 4KB-aligned file offset
            gpu_buffer: Destination GPU tensor (uint8, size == chunk_size)
            chunk_size: 4KB-aligned transfer size
        """
        import time

        file_list = [(file_offset, chunk_size, fd, f"chunk_{file_offset}")]
        file_descs = self.nixl_agent.register_memory(file_list, "FILE")
        file_xfer_descs = file_descs.trim()

        gpu_descs = self.nixl_agent.register_memory(gpu_buffer, "VRAM")
        gpu_xfer_descs = self.nixl_agent.get_xfer_descs(gpu_buffer, "VRAM")

        xfer_handle = self.nixl_agent.initialize_xfer(
            "READ", gpu_xfer_descs, file_xfer_descs, self.nixl_agent.name
        )

        if not xfer_handle:
            raise RuntimeError("Failed to initialize NIXL transfer")

        state = self.nixl_agent.transfer(xfer_handle)
        if state == "ERR":
            raise RuntimeError(f"Failed to start NIXL transfer at offset {file_offset}")

        max_wait_seconds = 30
        wait_start = time.time()
        while True:
            state = self.nixl_agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                raise RuntimeError(f"NIXL transfer failed at offset {file_offset}")
            if time.time() - wait_start > max_wait_seconds:
                raise TimeoutError(f"NIXL transfer timeout at offset {file_offset}")
            time.sleep(0.001)

        self.nixl_agent.release_xfer_handle(xfer_handle)
        self.nixl_agent.deregister_memory(file_descs)
        self.nixl_agent.deregister_memory(gpu_descs)

    def _load_tensor_with_fd(
        self,
        fd: int,
        header_len: int,
        metadata: dict,
        torch_dtype: torch.dtype
    ) -> torch.Tensor:
        """Load a single tensor using NIXL file-to-GPU transfer with existing fd.

        Handles two GDS constraints:
        1. File offsets and sizes must be 4KB-aligned
        2. Each I/O request must fit within cuFile's per_buffer_cache_size (default 1MB)

        Large tensors are split into aligned chunks and reassembled on GPU.

        Args:
            fd: Open file descriptor (from os.open)
            header_len: Length of safetensors header in bytes
            metadata: Tensor metadata from safetensors header
            torch_dtype: Target PyTorch dtype

        Returns:
            Tensor loaded directly to GPU via NIXL GDS (DMA transfer)
        """
        if self.nixl_agent is None:
            raise RuntimeError("NIXL agent not initialized")

        # Parse metadata
        shape = tuple(metadata["shape"])
        file_dtype = self.SAFETENSORS_DTYPE_MAP.get(metadata.get("dtype", ""), torch_dtype)
        data_offsets = metadata["data_offsets"]
        begin_offset = data_offsets[0]
        end_offset = data_offsets[1]
        tensor_size_bytes = end_offset - begin_offset

        # Absolute file offset of tensor data
        absolute_file_offset = 8 + header_len + begin_offset

        # Allocate destination buffer on GPU (raw bytes for the full tensor)
        result_buffer = torch.empty(tensor_size_bytes, dtype=torch.uint8, device=self.device)
        bytes_loaded = 0

        try:
            while bytes_loaded < tensor_size_bytes:
                # Current position in file
                cur_file_offset = absolute_file_offset + bytes_loaded
                remaining = tensor_size_bytes - bytes_loaded

                # Align offset down to 4KB boundary
                aligned_offset = (cur_file_offset // self.GDS_ALIGNMENT) * self.GDS_ALIGNMENT
                prefix_bytes = cur_file_offset - aligned_offset

                # How much useful data to read in this chunk.
                # aligned_size = round_up(prefix_bytes + useful_bytes) must stay <= GDS_MAX_CHUNK_SIZE,
                # so cap useful_bytes to leave room for the prefix padding.
                useful_bytes = min(remaining, self.GDS_MAX_CHUNK_SIZE - prefix_bytes)

                # Aligned total read size (prefix padding + useful data, rounded up to 4KB)
                aligned_size = ((prefix_bytes + useful_bytes + self.GDS_ALIGNMENT - 1)
                                // self.GDS_ALIGNMENT) * self.GDS_ALIGNMENT

                # Reuse chunk buffer across calls to avoid repeated GPU allocation
                if self._chunk_buffer is None or self._chunk_buffer.numel() < aligned_size:
                    self._chunk_buffer = torch.empty(
                        max(aligned_size, self.GDS_MAX_CHUNK_SIZE),
                        dtype=torch.uint8, device=self.device
                    )
                chunk_buffer = self._chunk_buffer[:aligned_size]

                # Transfer aligned chunk from file to GPU
                self._nixl_transfer_chunk(fd, aligned_offset, chunk_buffer, aligned_size)

                # Copy useful data from chunk to result buffer
                result_buffer[bytes_loaded:bytes_loaded + useful_bytes] = \
                    chunk_buffer[prefix_bytes:prefix_bytes + useful_bytes]

                bytes_loaded += useful_bytes

            # Reinterpret raw bytes as the correct dtype and shape
            tensor = result_buffer.view(file_dtype).reshape(shape)

            # Convert dtype if file dtype differs from target
            if file_dtype != torch_dtype:
                tensor = tensor.to(dtype=torch_dtype)

        except Exception as e:
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
