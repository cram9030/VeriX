# VeriX Docker Environment

This directory contains Docker configuration files for running VeriX in a containerized environment with GPU support.

## Files in This Directory

- **Dockerfile**: Multi-stage build for VeriX with Marabou and GPU support
- **docker-compose.yml**: Docker Compose configuration for easy container orchestration
- **test_environment.py**: Validation script to verify all dependencies
- **QUICKSTART.md**: Quick start guide with common commands and troubleshooting
- **.dockerignore**: Files to exclude from Docker build context

## Quick Start

See [QUICKSTART.md](./QUICKSTART.md) for detailed instructions.

### TL;DR

```bash
# Build
docker-compose build verix-gpu

# Test
docker-compose run --rm verix-gpu python3 test_environment.py

# Run MNIST example
docker-compose run --rm verix-gpu bash -c "cd /workspace/verix && python3 mnist.py"
```

## Architecture

### Multi-Stage Build

The Dockerfile uses a two-stage build process:

1. **Stage 1 (marabou-builder)**:
   - Based on Ubuntu 24.04
   - Compiles Marabou from source with Gurobi support
   - Installs Gurobi 11.0.3
   - Build artifacts are cached for faster subsequent builds

2. **Stage 2 (runtime)**:
   - Based on **NVIDIA NGC TensorFlow container** (`nvcr.io/nvidia/tensorflow:25.01-tf2-py3`)
   - Includes pre-installed: Python 3.10, TensorFlow 2.x, CUDA 12.6, cuDNN
   - Copies built Marabou and Gurobi from Stage 1
   - Installs additional dependencies (ONNX, Keras 3.x, scikit-image, etc.)
   - No PEP 668 pip restrictions

### Why Multi-Stage?

- **Smaller final image**: Build tools aren't included in runtime
- **Faster rebuilds**: Marabou compilation (~20 min) is cached
- **Cleaner separation**: Build vs. runtime dependencies

### Why NVIDIA NGC TensorFlow Container?

The runtime stage uses NVIDIA's NGC TensorFlow container instead of building from scratch:

- **Pre-optimized**: TensorFlow and CUDA libraries are tuned for NVIDIA GPUs
- **No pip issues**: Designed for pip installations, no `--break-system-packages` required
- **Faster builds**: TensorFlow already installed (~10-15 min vs 30+ min)
- **Regular updates**: NVIDIA releases monthly updates with security fixes and performance improvements
- **Battle-tested**: Used in production ML workloads worldwide
- **Smaller images**: More efficient than building TensorFlow from source

## Configuration

### GPU Support

The container uses NVIDIA's CUDA base image and requires:
- NVIDIA GPU with compute capability 5.0+
- NVIDIA drivers 450.80.02+
- NVIDIA Container Toolkit

GPU access is configured in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, compute, utility]
```

### CPU-Only Mode

For systems without GPU, use the `verix-cpu` service:
```bash
docker-compose run --rm verix-cpu
```

Note: This uses the same image but without GPU device allocation.

## Library Versions

### Current (Docker Environment)

| Library | Version | Notes |
|---------|---------|-------|
| Base Image | nvcr.io/nvidia/tensorflow:24.05-tf2-py3 | NVIDIA NGC optimized container |
| Python | 3.10.6 | Compatible with Maraboupy (requires >=3.8, <=3.12) |
| TensorFlow | 2.x | Pre-installed by NVIDIA, with CUDA 12.4 |
| Keras | 3.3.3 | Upgraded from NGC default |
| ONNX | 1.16.0 | Latest stable |
| ONNXRuntime | 1.17.3 | GPU-enabled |
| Marabou | Latest | Built from source |
| Gurobi | 11.0.3 | License required |

### Original (VeriX Paper)

| Library | Version |
|---------|---------|
| Python | 3.7.13 |
| TensorFlow | 2.9.1 |
| Keras | 2.9.0 |
| ONNX | 1.10.2 |
| ONNXRuntime | 1.10.0 |

### Known Compatibility Notes

1. **NVIDIA NGC Base**: Using optimized TensorFlow from NVIDIA
   - Pre-configured for GPU workloads
   - No PEP 668 pip restrictions
   - Monthly security and performance updates

2. **Keras 3.x API Changes**: Some imports may need updating
   - Old: `from keras.layers import Dense`
   - New: `from keras import layers; layers.Dense(...)`
   - NGC container includes Keras 2.x; we upgrade to 3.x for compatibility

3. **Python 3.10**: Stable version used by NGC containers
   - Compatible with all VeriX requirements
   - No breaking changes from original 3.7

## Customization

### Changing Python Versions

To use a different Python version, modify the Dockerfile:

```dockerfile
# Stage 2
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu24.04

# Install specific Python version
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.11 python3.11-dev
```

### Adjusting Marabou Build Options

In the Dockerfile, modify the CMake configuration:

```dockerfile
RUN mkdir build && cd build && \
    cmake .. \
    -DENABLE_GUROBI=OFF \        # Disable Gurobi
    -DENABLE_OPENBLAS=ON \
    -DBUILD_PYTHON=ON \
    -DRUN_UNIT_TEST=ON \         # Enable unit tests
    ...
```

### Using Different CUDA Versions

Change the base image in Stage 2:

```dockerfile
# For CUDA 12.4 with Ubuntu 22.04
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

# Then adjust TensorFlow version accordingly
RUN python3 -m pip install tensorflow[and-cuda]==2.16.1
```

### Adding More Python Packages

Add to the pip install command in Dockerfile:

```dockerfile
RUN python3 -m pip install \
    tensorflow[and-cuda]==2.16.1 \
    keras==3.3.3 \
    your-package-here \
    ...
```

## Volume Mounts

Default mounts (configured in docker-compose.yml):

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `../` | `/workspace/verix` | VeriX source code (read-write) |
| `../models` | `/workspace/verix/models` | Model files (read-write) |
| `./outputs` | `/workspace/outputs` | Generated outputs (read-write) |
| `~/gurobi.lic` | `/opt/gurobi/gurobi.lic` | Gurobi license (read-only, optional) |

Changes to files in mounted directories are immediately reflected in the container.

## Environment Variables

Key environment variables set in the container:

### CUDA/GPU
- `NVIDIA_VISIBLE_DEVICES=all`: Makes all GPUs visible
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: Required CUDA capabilities

### TensorFlow
- `TF_CPP_MIN_LOG_LEVEL=2`: Reduce TF logging verbosity
- `TF_ENABLE_ONEDNN_OPTS=0`: Disable oneDNN optimizations (can cause issues)
- `TF_FORCE_GPU_ALLOW_GROWTH=true`: Allocate GPU memory as needed

### Gurobi
- `GUROBI_HOME=/opt/gurobi/linux64`: Gurobi installation path
- `GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic`: License file location

### Python
- `PYTHONUNBUFFERED=1`: Force unbuffered output
- `PYTHONDONTWRITEBYTECODE=1`: Don't create .pyc files

## Building Without Cache

To force a complete rebuild:

```bash
docker-compose build --no-cache verix-gpu
```

This is useful when:
- Marabou source has been updated
- Dependency versions have changed
- Build artifacts are corrupted

## Resource Requirements

### Minimum

- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 20 GB
- **GPU**: None (CPU mode works)

### Recommended

- **CPU**: 8+ cores (for parallel Marabou queries)
- **RAM**: 16 GB
- **Disk**: 50 GB (for models and datasets)
- **GPU**: NVIDIA GPU with 8+ GB VRAM

### Build Requirements

- **RAM**: 8 GB (for Marabou compilation)
- **Disk**: 10 GB free space
- **Time**: 20-30 minutes (first build)

## Troubleshooting

See [QUICKSTART.md](./QUICKSTART.md#troubleshooting) for common issues and solutions.

### Build Failures

If the build fails:
1. Check available disk space: `df -h`
2. Check Docker memory limit in settings
3. Try building with fewer parallel jobs (edit Dockerfile `-j` flag)
4. Check Docker logs: `docker-compose logs`

### Runtime Issues

If containers fail to start:
1. Check logs: `docker-compose logs verix-gpu`
2. Verify GPU access (if using GPU): `nvidia-smi`
3. Check mounted volumes exist
4. Verify Docker Compose syntax: `docker-compose config`

## Maintenance

### Updating Dependencies

To update Python packages, modify Dockerfile and rebuild:

```dockerfile
RUN python3 -m pip install \
    tensorflow[and-cuda]==2.17.0 \  # Updated version
    ...
```

Then:
```bash
docker-compose build --no-cache verix-gpu
```

### Updating Marabou

To use a specific Marabou version, modify Dockerfile:

```dockerfile
WORKDIR /opt
RUN git clone https://github.com/NeuralNetworkVerification/Marabou.git && \
    cd Marabou && \
    git checkout v2.0.0  # Specific version tag
```

## Contributing

When modifying the Docker setup:

1. Test both GPU and CPU variants
2. Run `test_environment.py` to verify all dependencies
3. Update this README with any configuration changes
4. Update QUICKSTART.md with new workflows
5. Document any breaking changes

## Additional Resources

- [VeriX Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/46907c2ff9fafd618095161d76461842-Paper-Conference.pdf)
- [Marabou Documentation](https://github.com/NeuralNetworkVerification/Marabou)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
