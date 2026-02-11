# 安装 NVIDIA GPUDirect Storage (GDS)

## 系统要求

- ✅ Ubuntu 20.04/22.04/24.04
- ✅ NVIDIA Driver ≥ 515
- ✅ CUDA ≥ 11.7
- ✅ NVMe SSD
- ✅ Root权限

你的系统：
- OS: Ubuntu 24.04 LTS ✓
- Driver: 570.211.01 ✓
- CUDA: 12.8 ✓
- GPU: RTX 6000 Ada ✓

## 快速安装

### 方法 1: 自动安装脚本

```bash
sudo bash scripts/install_gds.sh
```

### 方法 2: 手动安装

```bash
# 1. 添加NVIDIA CUDA仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 2. 安装GDS
sudo apt-get install -y nvidia-gds

# 3. 加载nvidia-fs驱动
sudo modprobe nvidia-fs

# 4. 验证
lsmod | grep nvidia_fs
```

如果看到 `nvidia_fs`，说明安装成功！

## 验证安装

### 1. 检查驱动

```bash
lsmod | grep nvidia_fs
```

**期望输出**：
```
nvidia_fs             331776  0
nvidia              62464000  183 nvidia_fs,...
```

### 2. 运行GDS检查工具

```bash
/usr/local/cuda/gds/tools/gdscheck -p
```

**期望输出**：
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

### 3. 测试真正的GDS

重新运行测试：

```bash
# 重新测试7B模型
python tests/compare.py --model 7b --loader nixl
```

**期望结果**（如果GDS工作）：
- 速度：应该更快（可能1.5-2×）
- CPU内存：应该节省（不增加1GB）

## 持久化配置

确保重启后自动加载：

```bash
# 创建配置文件
echo "nvidia-fs" | sudo tee /etc/modules-load.d/nvidia-fs.conf

# 或添加到 /etc/modules
echo "nvidia-fs" | sudo tee -a /etc/modules
```

## 常见问题

### Q1: 安装后仍然没有nvidia-fs？

```bash
# 检查内核模块路径
ls /lib/modules/$(uname -r)/updates/dkms/ | grep nvidia

# 手动加载
sudo depmod -a
sudo modprobe nvidia-fs
```

### Q2: "modprobe: FATAL: Module nvidia-fs not found"

可能原因：
1. **GDS包未正确安装**
   ```bash
   dpkg -l | grep nvidia-gds
   ```

2. **内核头文件缺失**（GDS需要编译DKMS模块）
   ```bash
   sudo apt-get install linux-headers-$(uname -r)
   sudo apt-get install --reinstall nvidia-gds
   ```

3. **驱动版本不兼容**
   - GDS需要 NVIDIA Driver ≥ 515
   - 你的570.211.01应该没问题

### Q3: 安装后CPU内存仍然很高？

1. **验证GDS真的在工作**：
   ```bash
   # 查看cuFile日志
   export CUFILE_ENV_PATH_JSON=/etc/cufile.json
   # 在cufile.json中启用logging
   ```

2. **检查文件系统支持**：
   ```bash
   df -Th models/qwen2.5-7b/
   ```
   - ext4/xfs: 支持 ✓
   - 其他: 可能不支持

3. **检查对齐**：
   - GDS要求4K对齐
   - safetensors文件应该满足要求

### Q4: 没有root权限怎么办？

如果无法安装nvidia-fs：
- ❌ 无法使用真正的GDS零拷贝
- ✅ NIXL会自动使用兼容模式
- ✅ 仍然比HF快（优化的POSIX I/O）
- ❌ 但CPU内存会增加

权衡：
- 有GDS：快1.5-2×，省CPU内存
- 无GDS：快1.4×，多用1GB CPU

## 测试对比

### 安装GDS前（兼容模式）

```
HF:   8.12s,  CPU +252MB
NIXL: 5.70s,  CPU +1287MB  (快1.42x)
```

### 安装GDS后（预期）

```
HF:   8.12s,  CPU +252MB
NIXL: 4-5s,   CPU +100-200MB  (快1.6-2x, 省CPU)
```

## 相关资源

- [NVIDIA GDS 文档](https://docs.nvidia.com/gpudirect-storage/)
- [cuFile API 参考](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
- [GDS 性能调优](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)

## 下一步

1. **安装GDS**：`sudo bash scripts/install_gds.sh`
2. **验证**：`lsmod | grep nvidia_fs`
3. **重新测试**：`python tests/compare.py --model 7b`
4. **对比结果**：CPU内存应该大幅降低

如果安装成功，NIXL GDS将展现真正的价值：**更快的速度 + 节省CPU内存**！
