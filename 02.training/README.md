# Model Training

This folder contains the main

---

## File Descriptions

- **`train.py`**  
  Main Python script for training a model using the ONNE Workflow
  Args:
    - s: Path to ONNE training data generated 
    - m (default: onne_model): Save path for model(s)
    - v (default: onne): Path to validation data, if empty, the scipt will generate a validation set using ONNE data generator
    - k (default: 10): Number of folds for k-fold cross-validation

  Example Usage:

    ```python
    python train.py -s ../00.data/trn_data/onne_ml_data.pkl

    python train.py -s ../00.data/trn_data/onne_ml_data.pkl -v ../00.data/trn_data/AIMD_data.pkl

    python train.py -s ../00.data/trn_data/onne_ml_data.pkl -v ../00.data/trn_data/AIMD_data.pkl -m models_onne -k 20
  ```

- **`md_train.py`**  
  Main Python script for training a model on chunks of AIMD/MLMD data 

  Args:
    - s: Path to ONNE training data generated 
    - m (default: onne_model): Save path for model(s)
    - k (default: 10): Number of folds for k-fold cross-validation

  Example Usage:

    ```python
    python md_train.py -s ../00.data/trn_data/onne_ml_data.pkl

    python md_train.py -s ../00.data/trn_data/AIMD_temperature_ml_data.pkl

    python md_train.py -s ../00.data/trn_data/onne_ml_data.pkl -m MD_trained -k 20
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