###########################
# Importing Libraries
###########################

# General libraries
import os, pickle, torch, argparse,numpy as np
from datetime import datetime
# Custom libraries
from training_tools import data_generator

# Machine learning libraries
from cnn_model import CNNModel
from train_model import train_model
from torch.utils.data import DataLoader, TensorDataset


#---------------------------------#
# Parsing arguments
#---------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str, default = None)
parser.add_argument("-m", type=str, default = "onne_model")
parser.add_argument("-v", type=str, default = "onne")
parser.add_argument("-k", type=int, default = 10)
args, _ = parser.parse_known_args()

#---------------------------------#
# Data Loading and pre-processing
#---------------------------------#

# Use GPU if available
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Programmed Started")
print("-------------------------------", flush = True)
device = torch.device("mps" if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu") # GPU
print(f"Program will use {device}", flush = True)

if device == torch.device("cuda"):
    print("GPU Name: ",torch.cuda.get_device_name(0), flush = True)
print("-------------------------------", flush = True)

model_dir = args.m
os.makedirs(model_dir, exist_ok=True)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Reading Data")
# Load preprocessed training data (macrostates) from a pickle file.
onne_data = pickle.load(open(args.s, "rb"))
onne_data = np.array([datapoint for datapoint in onne_data], dtype = object) # filter  --> if np.all(datapoint[1][:10] == 0)

# Count total number of training samples
num_of_datapoints = len(onne_data)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Data size after filtering: {num_of_datapoints}")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Reading Validation Data")

if args.v == "onne":
    x_val, y_val = next(data_generator(20000,num_of_datapoints,onne_data))
else:
    val_data = pickle.load(open(args.v,"rb"))
    val_data = np.array([datapoint for datapoint in val_data], dtype = object) # filter --> if np.all(datapoint[1][:10] == 0)
    x_val, y_val = np.vstack(val_data[:,0]), np.vstack(val_data[:,1])

x_val, y_val = torch.from_numpy(x_val).to(torch.float32), torch.from_numpy(y_val).to(torch.float32)
#  Wrap validation tensors into a PyTorch `TensorDataset` for batching
validation_dataset = TensorDataset(x_val, y_val)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Generated Val dataset from ONNE")

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Done")
print("-------------------------------", flush = True)

#---------------------------------#
# Hyperparameters
#---------------------------------#
epochs, batch_size, learning_rate, weight_decay = 50, 128, 0.000635, 0.000052
batches_per_epoch = 780
# Initialize custom data generator for training
train_generator = data_generator(batch_size = batch_size,
                                 num_of_data_samples = num_of_datapoints,
                                 data = onne_data)

# Create a PyTorch DataLoader to serve batches during validation
val_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)

output_layer = len(y_val[0])
#%%
#---------------------------------#
# Model - Training
#---------------------------------#

print("-------------------------------", flush = True)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Beginning Model Training", flush = True)
print("-------------------------------", flush = True)

# Loop over K-folds (in this case, 10-fold training)
for fold in range(args.k): #key,model in models.items():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Training Fold {fold + 1}")
    print("-------------------------------", flush = True)
    
    # Instantiate a new model for each fold
    model = CNNModel(output_layer, p = 0.4) 
    
    # Define path for saving the model checkpoint for this fold
    save_path = os.path.join(model_dir, f"onne_{fold+1}.pth")
    
    # Train the model for this fold
    trained_model, history= train_model(
                model = model,
                epochs = epochs,
                learning_rate = learning_rate,
                train_generator = train_generator,
                val_loader = val_dataloader,
                device = device,
                steps_per_epoch = batches_per_epoch,
                weight_decay = weight_decay,
                early_stop = True
    )
    # Save model checkpoint, training history, and hyperparameters
    ckpt = {
        "model_state_dict": trained_model.state_dict(),  # Trained model weights
        "history": history,                         # Training and validation loss history
        "hyperparameters": { 
            "epochs":epochs,                    # Record of hyperparameters used
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "L2 Regularization": weight_decay,
            "iterations": batches_per_epoch,
            "Dropout" : 0.4
        }
    }
    
    # Save the checkpoint to disk
    torch.save(ckpt, save_path)
    print("---------------------------------------------------------", flush = True)
print("---------------------------------------------------------", flush = True)    
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Program Finished", flush = True)

