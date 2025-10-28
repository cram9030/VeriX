# VeriX: Towards Verified Explainability of Deep Neural Networks

**Min Wu, Haoze Wu, Clark Barrett.**

The accompanying paper [VeriX: towards Verified eXplainability of deep neural networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/46907c2ff9fafd618095161d76461842-Paper-Conference.pdf) is accepted by NeurIPS 2023.

#### Citation
```
@inproceedings{VeriX,
 author = {Wu, Min and Wu, Haoze and Barrett, Clark},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {22247--22268},
 publisher = {Curran Associates, Inc.},
 title = {VeriX: Towards Verified Explainability of Deep Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/46907c2ff9fafd618095161d76461842-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

## Example Usage

For the `MNIST` dataset, to compute the VeriX explanation for the `10`th image in the test set `x_test` and the neural network `mnist-10x2.onnx` in folder `models/`.

```
verix = VeriX(dataset="MNIST",
              image=x_test[10],
              model_path="models/mnist-10x2.onnx")
verix.traversal_order(traverse="heuristic")
verix.get_explanation(epsilon=0.05)
```
Use the `heuristic` feature-level sensitivity method to set the traversal order, and then set the perturbation magnitude `epsilon` to obtain the explanation. Be default, the *original image*, *the sensitivity*, and *the explanation* will be plotted and saved.

See `mnist.py` for a full example usage. The `GTSRB` dataset is also supported as in `gtsrb.py`. 


#### To use VeriX, a neural network verification tool called Marabou and an LP solver called Gurobi need to be installed in advance.
```
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd path/to/marabou/repo/folder
mkdir build 
cd build
cmake .. -DENABLE_GUROBI=ON -DBUILD_PYTHON=ON
cmake --build . -j 12
```
More details on how to install Marabou with Gurobi enabled can be found [here](https://github.com/NeuralNetworkVerification/Marabou).

## Docker Setup (Recommended)

A Docker environment is provided for easy setup with GPU support. This is the recommended way to get started with VeriX.

### Prerequisites

- **Docker** (version 20.10 or later)
- **Docker Compose** (version 1.29 or later)
- **NVIDIA Container Toolkit** (for GPU support) - [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Gurobi License** (optional, but recommended for better performance) - [Free Academic License](https://www.gurobi.com/academia/academic-program-and-licenses/)

### Building the Docker Image

Navigate to the `environment/` directory and build the image:

```bash
cd environment/
docker-compose build verix-gpu
```

For CPU-only systems:
```bash
docker-compose build verix-cpu
```

Build time: Approximately 20-30 minutes (Marabou compilation takes most of the time, but is cached for future builds).

### Running the Container

#### With GPU Support

```bash
docker-compose run --rm verix-gpu
```

This will start a bash shell inside the container with:
- VeriX source code mounted at `/workspace/verix`
- GPU access enabled
- All dependencies pre-installed

#### CPU-Only Mode

```bash
docker-compose run --rm verix-cpu
```

### Configuring Gurobi License

If you have a Gurobi license file:

1. Edit `environment/docker-compose.yml` and uncomment the license volume mount line:
   ```yaml
   volumes:
     - ~/gurobi.lic:/opt/gurobi/gurobi.lic:ro
   ```
2. Update `~/gurobi.lic` to point to your actual license file location

Alternatively, set the license inside the container by setting `GRB_LICENSE_FILE` environment variable.

### Common Docker Commands

#### Run the MNIST Example

```bash
# Start container
docker-compose run --rm verix-gpu

# Inside container
cd /workspace/verix
python3 mnist.py
```

#### Run the GTSRB Example

```bash
# Inside container
cd /workspace/verix
python3 gtsrb.py
```

#### Run Environment Tests

To verify all dependencies are working:

```bash
# Inside container
python3 test_environment.py
```

#### Interactive Development

Start a persistent container for development:

```bash
# Start container in background
docker-compose up -d verix-gpu

# Attach to running container
docker-compose exec verix-gpu bash

# When done, stop container
docker-compose down
```

#### Check GPU Availability

```bash
# Inside container
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Volume Mounts

The docker-compose configuration mounts:
- `../` → `/workspace/verix` (VeriX source code)
- `../models` → `/workspace/verix/models` (model files)
- `./outputs` → `/workspace/outputs` (results and outputs)

All changes to source code are reflected immediately in the container.

### Docker Environment Specifications

The Docker container is based on **NVIDIA NGC TensorFlow container** and includes:
- **Base Image**: nvcr.io/nvidia/tensorflow:24.05-tf2-py3
- **Base OS**: Ubuntu 22.04 LTS
- **Python**: 3.10.6 (compatible with Maraboupy requirement: >=3.8, <=3.12)
- **CUDA**: 12.4 with cuDNN (optimized by NVIDIA)
- **TensorFlow**: 2.x (pre-installed and optimized)
- **Keras**: 3.3.3
- **ONNX**: 1.16.0
- **ONNXRuntime**: 1.17.3 with GPU support
- **Marabou**: Latest version from source with Gurobi support
- **Additional**: numpy, matplotlib, scikit-image, tf2onnx

**Why NVIDIA NGC?** These containers are optimized for ML workloads, regularly updated, and include all necessary GPU libraries pre-configured.

### Troubleshooting

#### GPU Not Detected

If TensorFlow doesn't detect your GPU:
1. Verify NVIDIA drivers are installed: `nvidia-smi` (on host)
2. Verify NVIDIA Container Toolkit is installed
3. Check Docker can access GPU: `docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi`

#### Gurobi License Issues

If you see Gurobi license errors:
- Ensure license file is mounted correctly
- Check `GRB_LICENSE_FILE` environment variable points to the license
- Note: Marabou will fall back to its built-in LP solver if Gurobi is unavailable

#### Library Compatibility Issues

The Docker environment uses the NVIDIA NGC TensorFlow container (Python 3.10, TensorFlow 2.x). If you encounter compatibility issues:
- The NGC container uses Keras integrated with TensorFlow
- VeriX.py may need minor updates for Keras 3.x API if upgrading Keras independently
- Some import statements may need adjustment (e.g., `from keras import layers` vs `from tensorflow.keras import layers`)
- The original VeriX code used older library versions - most code should work, but test thoroughly

#### Out of Memory Errors

If you encounter memory issues:
- Increase Docker's memory limit in Docker Desktop settings
- Adjust `shm_size` in docker-compose.yml (default: 8GB for GPU, 4GB for CPU)
- Reduce batch sizes or model complexity

## Developer's Platform (Original Reference)

This was the original development environment. Docker is now recommended for easier setup.

```
python 		3.7.13
keras		2.9.0
tensorflow 	2.9.1
onnx 		1.10.2
onnxruntime 	1.10.0
tf2onnx 	1.9.3
```

**Note**: The Docker environment uses updated versions of these libraries (Python 3.12, TensorFlow 2.16, etc.) for better compatibility and performance.

## Remark

Thanks a lot for your interest in our work. Any questions please feel free to contact us: minwu@cs.stanford.edu.


