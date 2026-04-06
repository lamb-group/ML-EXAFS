# Hyperparameter tuning

This repository contains scripts and configuration files used to generate local atomic environments using the Objective Neural Network for EXAFS (ONNE) framework and to run FEFF simulations on generated structures.

---

## File Descriptions

- **`model_tuning.py`**  
  Python script for hyperparameter tuning using Random Search
  Args:
    - s: Path to ONNE training data generated 
    - v (default: onne): Path to validation data, if empty, the scipt will generate a validation set using ONNE data generator

  Example Usage:

    ```python
    python model_tuning.py -s ../00.data/trn_data/onne_ml_data.pkl

    python model_tuning.py -s ../00.data/trn_data/onne_ml_data.pkl -v ../00.data/trn_data/AIMD_data.pkl
  ```

- **`cnn_model.py`**  
  Class file containing the CNN architecture

- **`train_model.py`**  
  Methods file containing custom training functions for ONNE, MD and Fine tuned model

- **`training_tools.py`**  
  Utility functions used in training, most importantly the data generator function

- **`earlystopping.py`**  
  File containing custom implementation of early stopping technique

- **`run_ML.sh.py`**  
  SLURM script to train model on an HPC cluster with GPUs

### Documentation

- **`README.md`**  
  This file. Provides documentation describing the repository structure, workflow, dependencies, and usage instructions.