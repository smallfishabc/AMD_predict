# AMD Protein Prediction on MI300X

This project performs massive-scale protein prediction leveraging AMD MI300X accelerators.

## Overview

This repository contains code and data for large-scale protein structure and function prediction optimized for AMD's MI300X GPU architecture.

## Hardware

- **GPU**: AMD MI300X
- **Optimization**: ROCm-optimized deep learning workflows

## Project Structure

```
├── GmaxWm82ISU_01_724_v2.1.protein.fa  # Protein sequence data
└── README.md                            # This file
```

## Data

- **GmaxWm82ISU_01_724_v2.1.protein.fa**: Protein sequence dataset for prediction

## Requirements

- ROCm (for MI300X support)
- Python 3.8+
- PyTorch with ROCm support
- BioPython (for FASTA file handling)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AMD_predict

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
pip install biopython
```

## Usage

### Quick Start with Docker (Recommended for DigitalOcean AMD)

```bash
# 1. Build Docker image
./docker_setup.sh

### Benchmark Results

**ESM2-3B Embeddings:**
- Time: ~1 hour (80,374 sequences)
- Output: 191 GB (FP16, per-token)
- Adaptive batching: 2-256 by length

**SimpleFold Structure Prediction:**
- SimpleFold-3B: ~23 days (1 GPU, num_steps=200)
- SimpleFold-700M: ~12 days (1 GPU, faster, 95% accuracy)
- With 2 GPUs: ~6-12 days
- Priority subset: Process 10K proteins in 2-3 days

See [simplefold_quick_reference.py](simplefold_quick_reference.py) for details.

### Cloud Testing

Quick test on DigitalOcean AMD (5 minutes):
```bash
./cloud_quick_test.sh
```

Full testing guide: [CLOUD_TESTING.md](CLOUD_TESTING.md)

# 2. Run container
docker-compose up -d amd-protein-predict
docker-compose exec amd-protein-predict /bin/bash

# 3. Inside container, test GPU
rocm-smi
python -c "import torch; print(torch.cuda.is_available())"

# 4. Generate embeddings
python esm2_embedding_estimate.py
```

### Local Setup (Conda)

```bash
# Activate environment
conda activate amd_protein

# Analyze FASTA file
python analyze_fasta.py

# Generate embeddings
# Coming soon...
```

## Performance

This project is optimized for AMD MI300X accelerators to handle large-scale protein prediction tasks efficiently.

## License

TBD

## Contact

For questions and collaboration, please open an issue.
