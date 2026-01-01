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

Coming soon...

## Performance

This project is optimized for AMD MI300X accelerators to handle large-scale protein prediction tasks efficiently.

## License

TBD

## Contact

For questions and collaboration, please open an issue.
