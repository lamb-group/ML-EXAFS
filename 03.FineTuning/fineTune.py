# %%
#---------------------------------#
# Importing Libraries
#---------------------------------#
# General libraries
import os, pickle, numpy as np, argparse, sys
from glob import glob
from datetime import datetime

# Machine learning libraries
import torch
from cnn_model import CNNModel
from train_model import fineTune_model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#---------------------------------#
# Parsing arguments
#---------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str, required = True, help="Path to Fine tuning data")
parser.add_argument("-b", type=str, required = True, help="Directory to pretrained models")
parser.add_argument("-m", type=str, default = "ft_model", help="Directory to save the output model")
parser.add_argument("-t", type=float, default = 0.2, help="Train/Test Split (default: 0.2)")
args, _ = parser.parse_known_args()

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Programmed Started")

# Use GPU if available
print("-------------------------------", flush = True)
device = torch.device("mps" if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu") # GPU
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Program will use {device}", flush = True)

if device == torch.device("cuda"):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"GPU Name: ", torch.cuda.get_device_name(0), flush = True)
print("-------------------------------", flush = True)


working_dir = args.m; os.makedirs(working_dir, exist_ok=True)

#---------------------------------#
# Data Loading and processing
#---------------------------------#
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Reading Data")

ml_data = pickle.load(open(args.s,"rb"))

features, labels = zip(*ml_data)
x, y = np.vstack(features).astype(np.float32), np.vstack(labels).astype(np.float32)
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = args.t)
x_train, x_val, y_train, y_val = torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(y_train), torch.from_numpy(y_val)
                   
val_dataset = TensorDataset(x_val,y_val)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Done")
print("-------------------------------", flush = True)


#---------------------------------#
# Model - Training
#---------------------------------#

epochs, batch_size, learning_rate, weight_decay = 50, 128, 0.000635, 0.000052
trn_dataset = TensorDataset(x_train,y_train)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle =True)

print("-------------------------------", flush = True)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Beginning Model Training", flush = True)
print("-------------------------------", flush = True)

output_layer = len(y_val[0])

onne_models = sorted(glob(os.path.join(args.b,"*.pth")))

trn_dataloader = DataLoader(trn_dataset, batch_size= batch_size, shuffle = True)

for fold, model in enumerate(onne_models):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Fine tuning ONNE model #{fold+1}")
    print("---------------------------------------", flush = True)
    # Instantiate a new model for each fold
    ckpt = torch.load(model)
    base_model = CNNModel(output_layer, p = 0.4)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.to(device)

    # Define path for saving the model checkpoint for this fold
    save_path = os.path.join(working_dir, f"ft_model_{fold+1}.pth")

    # Train the model
    trained_model, history= fineTune_model(
                        model = base_model,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        train_loader=trn_dataloader,
                        val_loader=val_dataloader,
                        device=device,
                        weight_decay=weight_decay,
                        early_stop=True)
    # Save the model
    ckpt = {
        "model_state_dict": trained_model.state_dict(),
        "loss": history,
    }
    
    # Save
    torch.save(ckpt, save_path)
    print(f"Done with {fold+1}-fold")
    print("---------------------------------------------------------", flush = True)
print("---------------------------------------------------------", flush = True)    
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Program Finished", flush = True)

