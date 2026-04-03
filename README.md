# Objective Neural Network for EXAFS (ONNE)

This repository provides a complete workflow for generating datasets for ML-EXAFS using Objective Neural Network for EXAFS (ONNE) method. The example provided reproduces the results presented in "On The Robustness and Transferability of Statistical Structure Sampling for Machine Learning EXAFS"

---

## 1. Repository Structure

```
├── 00.data
│   ├── onne_main.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── run_feff_slurm.sh
│   ├── run_onne.sh
│   ├── Na.xyz
│   ├── Zr.xyz
│   ├── F.xyz
│   └── generated_configs/
├── 01.hyperParameterTuning
├── 02.training
├── 03.testing
├── logs/
└── README.md

```

---
## 2. Dependencies and Installation

### 2.1 Dependencies
- Linux (recommended)
- Python >= 3.10
- Conda or Miniconda (recommended)

### 2.2 Installation
```bash
git clone https://github.com/lamb-group/ML-EXAFS.git
cd ML-EXAFS

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ml-exafs

# Install package in development mode
pip install -e .
```
