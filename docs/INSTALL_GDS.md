# Installing NVIDIA GPUDirect Storage (GDS)

## System Requirements

- Ubuntu 20.04/22.04/24.04
- NVIDIA Driver >= 515
- CUDA >= 11.7
- NVMe SSD
- Root access

Our system:
- OS: Ubuntu 24.04 LTS
- Driver: 570.211.01
- CUDA: 12.8
- GPU: RTX 6000 Ada

## Quick Install

### Method 1: Automated Script

```bash
sudo bash scripts/install_gds.sh
```

### Method 2: Manual Installation

```bash
# 1. Add NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 2. Install GDS
sudo apt-get install -y nvidia-gds

# 3. Load nvidia-fs driver
sudo modprobe nvidia-fs

# 4. Verify
lsmod | grep nvidia_fs
```

If you see `nvidia_fs`, the installation was successful!

## Verifying Installation

### 1. Check Driver

```bash
lsmod | grep nvidia_fs
```

**Expected output**:
```
nvidia_fs             331776  0
nvidia              62464000  183 nvidia_fs,...
```

### 2. Run GDS Check Tool

```bash
/usr/local/cuda/gds/tools/gdscheck -p
```

**Expected output**:
```
GDS release version: x.x.x.x
nvidia_fs version:  x.x.x libcufile version: x.x
Platform: x86_64
...
CONFIGURATION:
NVMe               : Supported
NVMeOF             : Unsupported
SCSI               : Unsupported
...
BAR1 Size Check    : PASS
```

### 3. Test True GDS

Re-run the tests:

```bash
# Re-test 7B model
python tests/compare.py --model 7b --loader nixl
```

**Expected results** (if GDS is working):
- Speed: should be faster (potentially 1.5-2x)
- CPU memory: should be lower (no extra 1GB increase)

## Persistent Configuration

Ensure the driver loads automatically after reboot:

```bash
# Create config file
echo "nvidia-fs" | sudo tee /etc/modules-load.d/nvidia-fs.conf

# Or append to /etc/modules
echo "nvidia-fs" | sudo tee -a /etc/modules
```

## Troubleshooting

### Q1: nvidia-fs still missing after installation?

```bash
# Check kernel module path
ls /lib/modules/$(uname -r)/updates/dkms/ | grep nvidia

# Manual load
sudo depmod -a
sudo modprobe nvidia-fs
```

### Q2: "modprobe: FATAL: Module nvidia-fs not found"

Possible causes:
1. **GDS package not properly installed**
   ```bash
   dpkg -l | grep nvidia-gds
   ```

2. **Missing kernel headers** (GDS needs to compile DKMS module)
   ```bash
   sudo apt-get install linux-headers-$(uname -r)
   sudo apt-get install --reinstall nvidia-gds
   ```

3. **Driver version incompatible**
   - GDS requires NVIDIA Driver >= 515
   - Your 570.211.01 should be fine

### Q3: CPU memory still high after installation?

1. **Verify GDS is actually working**:
   ```bash
   # Check cuFile logs
   export CUFILE_ENV_PATH_JSON=/etc/cufile.json
   # Enable logging in cufile.json
   ```

2. **Check filesystem support**:
   ```bash
   df -Th models/qwen2.5-7b/
   ```
   - ext4/xfs: Supported
   - Others: May not be supported

3. **Check alignment**:
   - GDS requires 4K alignment
   - SafeTensors files should meet this requirement

### Q4: No root access?

If you cannot install nvidia-fs:
- Cannot use true GDS zero-copy
- NIXL will automatically use compatibility mode
- Still faster than HF (optimized POSIX I/O)
- But CPU memory will increase

Trade-offs:
- With GDS: 1.5-2x faster, saves CPU memory
- Without GDS: 1.4x faster, uses ~1GB more CPU memory

## Benchmark Comparison

### Before GDS Installation (Compatibility Mode)

```
HF:   8.12s,  CPU +252MB
NIXL: 5.70s,  CPU +1287MB  (1.42x faster)
```

### After GDS Installation (Expected)

```
HF:   8.12s,  CPU +252MB
NIXL: 4-5s,   CPU +100-200MB  (1.6-2x faster, saves CPU)
```

## Resources

- [NVIDIA GDS Documentation](https://docs.nvidia.com/gpudirect-storage/)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
- [GDS Performance Tuning](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)

## Next Steps

1. **Install GDS**: `sudo bash scripts/install_gds.sh`
2. **Verify**: `lsmod | grep nvidia_fs`
3. **Re-test**: `python tests/compare.py --model 7b`
4. **Compare results**: CPU memory should drop significantly

If installation succeeds, NIXL GDS will show its true value: **faster speed + lower CPU memory usage**!
