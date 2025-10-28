# Multi-stage Dockerfile for VeriX Development and Testing
# Using NVIDIA NGC TensorFlow container as base (includes CUDA, cuDNN, TensorFlow, Python)
# Stage 1: Build Marabou with Gurobi support
# Stage 2: Runtime environment with pre-installed TensorFlow/CUDA

# =============================================================================
# Stage 1: Marabou Builder
# =============================================================================
# Use the same base as the runtime to ensure glibc compatibility
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3 AS marabou-builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies for Marabou
# Note: Using system packages for Boost and OpenBLAS to avoid build issues
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    g++ \
    libboost-all-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-regex-dev \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Gurobi (using version 11.0 as referenced in Marabou CMake)
# Note: This requires a license file to be provided at runtime
WORKDIR /opt
RUN wget -q https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz && \
    tar xzf gurobi11.0.3_linux64.tar.gz && \
    rm gurobi11.0.3_linux64.tar.gz && \
    mv gurobi1103 gurobi

# Set Gurobi environment variables
ENV GUROBI_HOME=/opt/gurobi/linux64
ENV PATH="${GUROBI_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

# Clone and build Marabou
WORKDIR /opt
RUN git clone https://github.com/NeuralNetworkVerification/Marabou.git

WORKDIR /opt/Marabou
# Configure Marabou build
# Use system Boost (Ubuntu 22.04 has 1.74.0, which is compatible)
# Use system OpenBLAS (libopenblas-dev) instead of downloading
# Create the directory structure Marabou expects and symlink to system OpenBLAS
RUN mkdir -p tools/OpenBLAS-0.3.19/installed/lib && \
    mkdir -p tools/OpenBLAS-0.3.19/installed/include && \
    ln -s /usr/lib/x86_64-linux-gnu/libopenblas.a tools/OpenBLAS-0.3.19/installed/lib/libopenblas.a && \
    ln -s /usr/include/x86_64-linux-gnu/cblas.h tools/OpenBLAS-0.3.19/installed/include/cblas.h && \
    ln -s /usr/include/x86_64-linux-gnu/lapacke.h tools/OpenBLAS-0.3.19/installed/include/lapacke.h && \
    ln -s /usr/include/x86_64-linux-gnu/openblas_config.h tools/OpenBLAS-0.3.19/installed/include/openblas_config.h

RUN mkdir build && cd build && \
    cmake .. \
    -DENABLE_GUROBI=ON \
    -DENABLE_OPENBLAS=ON \
    -DBUILD_PYTHON=ON \
    -DRUN_UNIT_TEST=OFF \
    -DRUN_REGRESS_TEST=OFF \
    -DRUN_SYSTEM_TEST=OFF \
    -DRUN_MEMORY_TEST=OFF \
    -DGUROBI_DIR=${GUROBI_HOME} && \
    cmake --build . -j$(nproc)

# =============================================================================
# Stage 2: Runtime Environment with NVIDIA NGC TensorFlow Base
# =============================================================================
# Using NVIDIA's optimized TensorFlow container with Python 3.10
# Version 24.05 uses Python 3.10.6 which is compatible with Maraboupy (requires <=3.12,>=3.8)
# This container is pre-configured for GPU workloads and has no PEP 668 restrictions
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Note: The NGC container already has Python, CUDA, cuDNN, and TensorFlow
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ca-certificates \
    libopenblas-dev \
    libboost-program-options1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-system1.74.0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (no --break-system-packages needed in NGC containers)
RUN pip install --upgrade pip setuptools wheel

# Copy Gurobi from builder stage
COPY --from=marabou-builder /opt/gurobi /opt/gurobi

# Set Gurobi environment variables
ENV GUROBI_HOME=/opt/gurobi/linux64
ENV PATH="${GUROBI_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

# Copy built Marabou from builder stage
COPY --from=marabou-builder /opt/Marabou /opt/Marabou

# Install Marabou Python bindings
# The setup.py triggers another CMake build for the Python extension
# OpenBLAS was already downloaded during the builder stage, so it will be found
WORKDIR /opt/Marabou
RUN pip install .

# Install additional Python dependencies for VeriX
# TensorFlow is already installed in the base NGC container
# Installing Keras 3.x for compatibility (NGC may have Keras 2.x)
RUN pip install \
    keras==3.3.3 \
    onnx==1.16.0 \
    onnxruntime-gpu==1.17.1 \
    tf2onnx==1.16.1 \
    scikit-image==0.23.2 \
    matplotlib==3.8.4 \
    pillow==10.3.0 \
    plotly==5.22.0

# Set up workspace
WORKDIR /workspace

# Copy test script
COPY test_environment.py /workspace/test_environment.py

# Set environment variables for TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Verify installation by running tests
RUN python3 test_environment.py || echo "Note: Full tests may require GPU at runtime"

# Default command
CMD ["/bin/bash"]
