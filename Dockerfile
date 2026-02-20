FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and GDS (libcufile) packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libcufile-12-8 \
    libcufile-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.12 as default python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Install core Python dependencies (layer caching)
# nixl is optional — don't fail the whole build if unavailable
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages \
    torch transformers safetensors accelerate \
    huggingface-hub numpy psutil tqdm \
    && pip install --no-cache-dir --break-system-packages \
    "nixl[cu12]>=0.4.0" || echo "WARN: nixl not installed, will use compat/fallback mode"

# Copy project source
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY scripts/ scripts/
COPY configs/ configs/
COPY README.md .

# Install the package
RUN pip install --no-cache-dir --break-system-packages .

# Models directory as mount point
VOLUME /app/models

# cuFile/GDS library path and symlink
ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
RUN ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0 \
           /usr/local/cuda/targets/x86_64-linux/lib/libcufile.so || true

# GDS config
ENV CUFILE_ENV_PATH_JSON=/app/configs/cufile.json
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python"]
CMD ["tests/test_nixl.py", "--model", "0.6b"]
