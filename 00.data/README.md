# Objective Neural Network for EXAFS (ONNE)

This repository contains scripts and configuration files used to generate local atomic environments using the Objective Neural Network for EXAFS (ONNE) framework and to run FEFF simulations on generated structures.


## File Descriptions

### Structure Files

The structure files are required by packmol for each atom. Here, NaF-ZrF<sub>4</sub> molten salt is constructed therefore, three file species are used. Must be in .xyz format

- **`F.xyz`**  
  Atom file for Fluorine.

- **`Na.xyz`**  
  Atom file for Sodium.

- **`Zr.xyz`**  
  Atom file for Zirconium.

---

### Python Scripts

- **`onne_main.py`**  
  Main script for generating ONNE configurations. It should be edited to reflect the statistics of the sampled configurations.

  Args:
    - nprocs (default = 1): Number of CPUs for parallelization
    - workdir (default = "./generated_configs"): Working directory to save the output data
    - scripts (default = 4): Number of FEFF scripts (Hardware dependent)

Example Usage:

  ```python
  python onne_main.py

  python onne_main.py --nprocs 10 --workdir sampled_structures
  ```

- **`preprocessing.py`**  
  Processes the output data to be ready for machine learning training.

  Args:
    - --workdir: Directory that contains ONNE generated structures (default: None)
    - --name: Output file name (default: ONNE)
    - --output : Working directory to save the output data (default: ./ml_data)

Example Usage:

  ```python
  python preprocessing.py --workdir sampled_structures

  python preprocessing.py --workdir sampled_structures --name sample1 --output trn_data
  ```

- **`utils.py`**  
  Utility functions used across the workflow for onne_main.py and preprocessing.py, including file handling and helper routines.

---

### Execution Scripts

- **`run_onne.sh`**  
  Example shell script used to execute the ONNE workflow on a SLURM compute node.

- **`run_feff_slurm.sh`**  
  Example SLURM batch script output from onne_main.py used to submit parallel FEFF calculations on HPC systems.
  SLURM parameters could be edited in utils.py
---

### Data
- **train_test_data.tar.gz** 

  Compressed archive containing:
  - **`trn_data/`**  
    Directory containing training data prepared using preprocessing.py for ONNE and AIMD data
  - **`test_data/`**  
    Directory containing testing data
---

### Documentation

- **`README.md`**  
  Provides documentation describing the repository structure, workflow, dependencies, and usage instructions.