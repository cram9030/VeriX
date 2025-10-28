# VeriX Docker Quick Start Guide

## Prerequisites Checklist

- [ ] Docker installed (20.10+)
- [ ] Docker Compose installed (1.29+)
- [ ] NVIDIA Container Toolkit installed (for GPU support)
- [ ] NVIDIA GPU with drivers installed (optional but recommended)
- [ ] Gurobi license file (optional but recommended)

## Quick Start (5 minutes)

### 1. Build the Container

```bash
cd environment/
docker-compose build verix-gpu
```

**Expected time**: 20-30 minutes (first build only)

### 2. Test the Environment

```bash
docker-compose run --rm verix-gpu python3 test_environment.py
```

You should see all tests pass with green checkmarks.

### 3. Run MNIST Example

```bash
docker-compose run --rm verix-gpu bash -c "cd /workspace/verix && python3 mnist.py"
```

This will:
- Load the MNIST test dataset
- Run VeriX explanation on the 10th test image
- Generate and save visualization images

## Common Workflows

### Interactive Development Session

```bash
# Start container
docker-compose run --rm verix-gpu

# You're now inside the container
cd /workspace/verix
python3 mnist.py

# Make changes to source files (they're live-mounted)
# Test changes immediately

# Exit when done
exit
```

### Run Single Command

```bash
docker-compose run --rm verix-gpu python3 /workspace/verix/gtsrb.py
```

### Check GPU Status

```bash
docker-compose run --rm verix-gpu nvidia-smi
docker-compose run --rm verix-gpu python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### CPU-Only Usage

If you don't have a GPU or want to use CPU:

```bash
docker-compose run --rm verix-cpu
```

## File Locations Inside Container

| Host Location | Container Location | Purpose |
|--------------|-------------------|---------|
| `../` (VeriX root) | `/workspace/verix` | Source code |
| `../models` | `/workspace/verix/models` | Model files |
| `./outputs` | `/workspace/outputs` | Generated outputs |
| Your Gurobi license | `/opt/gurobi/gurobi.lic` | Gurobi license |

## Verifying Your Setup

### Step 1: Check Python and Imports

```bash
docker-compose run --rm verix-gpu python3 -c "
import tensorflow as tf
import keras
from maraboupy import Marabou
import onnx
import onnxruntime
print('All imports successful!')
print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)
"
```

### Step 2: Check GPU Access

```bash
docker-compose run --rm verix-gpu python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✓ {len(gpus)} GPU(s) detected')
    for gpu in gpus:
        print(f'  - {gpu.name}')
else:
    print('⚠ No GPU detected (CPU mode)')
"
```

### Step 3: Test Marabou

```bash
docker-compose run --rm verix-gpu python3 -c "
from maraboupy import Marabou
options = Marabou.createOptions(numWorkers=4)
print('✓ Marabou working')
"
```

## Configuring Gurobi (Optional)

### Option 1: Mount License File

Edit `docker-compose.yml` and uncomment:

```yaml
volumes:
  - ~/gurobi.lic:/opt/gurobi/gurobi.lic:ro  # Uncomment this line
```

Change `~/gurobi.lic` to your actual license path.

### Option 2: Web License Service (WLS)

If using Gurobi WLS, set environment variables in `docker-compose.yml`:

```yaml
environment:
  - GUROBI_WLS_ACCESSID=your-access-id
  - GUROBI_WLS_SECRET=your-secret
  - GUROBI_WLS_LICENSESERIAL=your-license
```

### Verify Gurobi

```bash
docker-compose run --rm verix-gpu python3 -c "
import gurobipy
print('Gurobi version:', gurobipy.gurobi.version())
"
```

## Troubleshooting

### Issue: "Error response from daemon: could not select device driver"

**Solution**: Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "No GPU detected" inside container

**Check 1**: Verify host GPU works:
```bash
nvidia-smi
```

**Check 2**: Test Docker GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi
```

**Check 3**: Ensure using `verix-gpu` not `verix-cpu` service.

### Issue: Out of memory errors

**Solutions**:
1. Increase Docker memory limit (Docker Desktop → Settings → Resources)
2. Increase shared memory in `docker-compose.yml`:
   ```yaml
   shm_size: '16gb'  # Increase from 8gb
   ```
3. Reduce batch size in your training code

### Issue: Gurobi license not found

**Check**:
```bash
docker-compose run --rm verix-gpu bash -c "ls -l /opt/gurobi/gurobi.lic"
```

If file doesn't exist, ensure volume mount is correct in `docker-compose.yml`.

### Issue: Build fails during Marabou compilation

**Solutions**:
1. Increase Docker memory (needs ~8GB for compilation)
2. Reduce parallel build jobs:
   ```dockerfile
   # In Dockerfile, change:
   cmake --build . -j4  # Instead of -j$(nproc)
   ```

### Issue: TensorFlow can't find CUDA libraries

This shouldn't happen with the provided Docker image. If it does:
1. Verify you're using `verix-gpu` service
2. Check CUDA is available: `docker-compose run --rm verix-gpu nvcc --version`

## Performance Tips

### GPU Memory Management

TensorFlow is configured with `TF_FORCE_GPU_ALLOW_GROWTH=true` to allocate GPU memory as needed. To use a different strategy, modify the environment variable in `docker-compose.yml`.

### Parallel Marabou Queries

VeriX uses Marabou options with `numWorkers=16` by default (see `VeriX.py:32`). Adjust based on your CPU:

```python
options = Marabou.createOptions(
    numWorkers=8,  # Adjust to your CPU core count
    timeoutInSeconds=300,
    verbosity=0,
    solveWithMILP=True
)
```

## Next Steps

1. Read the main README for VeriX usage details
2. Explore `mnist.py` and `gtsrb.py` examples
3. Try modifying epsilon values to see different explanations
4. Check the [VeriX paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/46907c2ff9fafd618095161d76461842-Paper-Conference.pdf) for methodology details