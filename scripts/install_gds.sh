#!/bin/bash
# Install NVIDIA GPUDirect Storage (GDS)

set -e

echo "=================================================="
echo "NVIDIA GPUDirect Storage Installation"
echo "=================================================="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (sudo)"
   exit 1
fi

# System info
echo "System: $(lsb_release -ds)"
echo "Kernel: $(uname -r)"
echo "NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo ""

# Step 1: Add NVIDIA CUDA repository
echo "Step 1: Adding NVIDIA CUDA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# Step 2: Install GDS packages
echo ""
echo "Step 2: Installing GDS packages..."
apt-get install -y nvidia-gds

# Step 3: Load nvidia-fs module
echo ""
echo "Step 3: Loading nvidia-fs kernel module..."
modprobe nvidia-fs

# Step 4: Verify installation
echo ""
echo "Step 4: Verifying installation..."
if lsmod | grep -q nvidia_fs; then
    echo "✓ nvidia-fs module loaded successfully"
else
    echo "✗ nvidia-fs module not loaded"
    exit 1
fi

# Step 5: Check GDS configuration
echo ""
echo "Step 5: Checking GDS configuration..."
if [ -f /usr/local/cuda/gds/tools/gdscheck ]; then
    /usr/local/cuda/gds/tools/gdscheck -p
else
    echo "⚠ gdscheck tool not found, but module is loaded"
fi

# Step 6: Create persistent configuration
echo ""
echo "Step 6: Creating persistent module loading..."
echo "nvidia-fs" > /etc/modules-load.d/nvidia-fs.conf

echo ""
echo "=================================================="
echo "✓ GDS Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Reboot to ensure nvidia-fs loads on boot"
echo "2. Or manually load: sudo modprobe nvidia-fs"
echo "3. Test with: lsmod | grep nvidia_fs"
echo ""
