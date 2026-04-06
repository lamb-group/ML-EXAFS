# General libraries
import os, pickle, numpy as np, argparse
from glob import glob
from datetime import datetime

# Machine learning libraries
import torch
# from torchinfo import summary
from cnn_model import CNNModel
from train_model import train_md_model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str, default = None)
parser.add_argument("-m", type=str, default = "md_model")
parser.add_argument("-k", type=int, default = 10)
args, _ = parser.parse_known_args()

# # Use GPU if available
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Programmed Started")
print("-------------------------------", flush = True)
device = torch.device("mps" if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu") # GPU
print(f"Program will use {device}", flush = True)
if device == torch.device("cuda"):
    print("GPU Name: ",torch.cuda.get_device_name(0), flush = True)
print("-------------------------------", flush = True)

working_dir = args.m; os.makedirs(working_dir, exist_ok=True)

#---------------------------------#
# Data Loading and processing
#---------------------------------#
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Reading Data")

ml_data = pickle.load(open(args.s,"rb"))#

x, y = zip(*ml_data)
x, y = np.stack(x).astype(np.float32), np.stack(y).astype(np.float32)
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size= 0.1)

x_train, x_val, y_train, y_val = torch.from_numpy(x_train).to(torch.float32),torch.from_numpy(x_val).to(torch.float32), torch.from_numpy(y_train).to(torch.float32), torch.from_numpy(y_val).to(torch.float32)
                   
train_dataset, validation_dataset = TensorDataset(x_train,y_train), TensorDataset(x_val,y_val)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Done")
print("-------------------------------", flush = True)

#---------------------------------#
# Model - Training
#---------------------------------#

epochs, batch_size, learning_rate, weight_decay = 50, 128, 0.000635, 0.000052
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True)
val_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)

print("-------------------------------", flush = True)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Beginning Model Training", flush = True)
print("-------------------------------", flush = True)
output_layer = len(y_val[0])

for chunk in [10, 50, 100, 500, 1000, 5000, 10000, 20000]:

    chunk_dir = os.path.join(working_dir,f"subset_{chunk}"); os.makedirs(chunk_dir, exist_ok = True)
    
    # choice = np.random.choice(x_train.shape[0], size = chunk ,replace = False)
    x_trn, y_trn = x_train[:chunk], y_train[:chunk]
    
    trn_dataset = TensorDataset(x_trn,y_trn)
    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, shuffle = True)
    
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Training with data size #{chunk}")
    print("---------------------------------------", flush = True)
    for fold in range(args.k):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Training fold #{fold+1}")
        print("---------------------------------------", flush = True)
        # Instantiate a new model for each fold
        model = CNNModel(output_layer, p = 0.4)  
        # Define path for saving the model checkpoint for this fold
        save_path = os.path.join(chunk_dir, f"MD_model_{fold+1}.pth")
        # Train the model
        trained_model, history= train_md_model(
                            model=model,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            train_loader=trn_dataloader,
                            val_loader=val_dataloader,
                            device=device,
                            weight_decay=weight_decay)
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

