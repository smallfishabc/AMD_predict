# AMD MI300X Protein Prediction Docker Image
# Optimized for ROCm with ESM2-3B and Apple SimpleFold

FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_VERSION=6.1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/rocm/bin:$PATH
ENV HF_HOME=/workspace/models/huggingface

LABEL maintainer="smallfishabc"
LABEL description="AMD MI300X optimized for ESM2-3B embeddings and SimpleFold structure prediction"
LABEL rocm.version="6.1"
LABEL simplefold.version="latest"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    cmake \
    libhdf5-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core scientific computing packages
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    h5py \
    tqdm

# Install bioinformatics packages
RUN pip install --no-cache-dir \
    biopython \
    biotite \
    pyhmmer \
    fair-esm

# Install ESM2 and related packages
RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    tokenizers \
    safetensors

# Install structure prediction dependencies
RUN pip install --no-cache-dir \
    dm-tree \
    ml-collections \
    einops \
    jax \
    py3Dmol \
    hydra-core \
    omegaconf

# Clone and install Apple SimpleFold
RUN git clone https://github.com/apple/ml-simplefold.git /tmp/ml-simplefold && \
    cd /tmp/ml-simplefold && \
    pip install -e . && \
    cd / && rm -rf /tmp/ml-simplefold

# Install ESM (required by SimpleFold)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/esm.git

# Create working directory
WORKDIR /workspace

# Copy project files
COPY requirements.txt /workspace/
COPY *.py /workspace/
COPY docker_setup.sh test_environment.sh /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/m 7777

# Default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD rocm-smi && python -c "import simplefold" || exit 1

# Build instructions:
# docker build -t amd-protein-predict:rocm6.1-simplefold -f Dockerfile .
#
# Run instructions:
# docker run --rm -it \
#   --device=/dev/kfd --device=/dev/dri \
#   --security-opt seccomp=unconfined \
#   --group-add video \
#   --shm-size=32g \
#   -v $(pwd):/workspace \
#   -v /path/to/data:/workspace/data \
#   amd-protein-predict:rocm6.1-simplefold
#
# Test SimpleFold installation:
# python /workspace/test_simplefold_install.py

# Build instructions:
# docker build -t amd-protein-predict:rocm6.1 -f Dockerfile .
#
# Run instructions:
# docker run --rm -it \
#   --device=/dev/kfd --device=/dev/dri \
#   --security-opt seccomp=unconfined \
#   --group-add video \
#   -v $(pwd):/workspace \
#   -v /path/to/data:/workspace/data \
#   amd-protein-predict:rocm6.1
