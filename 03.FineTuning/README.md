# Model Training

---

## File Descriptions

- **`fineTune.py`**  
  Main Python script for fineTuning a pretrained ONNE model
  Args:
    - s: Path to ONNE training data generated 
    - b: Directory to pretrained models
    - m (default: ft_model): Directory to save the model
    - t (default: 0.2): Train/Test Split

  Example Usage:

    ```python
    python fineTune.py -s ../00.data/trn_data/AIMD_ml_data.pkl -b ../01.training/onne_models
  ```

- **`cnn_model.py`**  
  Class file containing the CNN architectures

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