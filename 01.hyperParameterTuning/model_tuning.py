#%%
# General libraries
import os, pickle, torch, numpy as np, argparse
from datetime import datetime

# Custom libraries
from training_tools import data_generator

# Machine learning libraries
from cnn_model import CNNModel
from train_model import train_model
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str, default = None)
parser.add_argument("-v", type=str, default = "onne")
args, _ = parser.parse_known_args()
onne_file, val = vars(args).values()
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Programmed Started")
print("-------------------------------", flush = True)

# Use GPU if available
device = torch.device("mps" if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu") # GPU

print(f"Program will use {device}", flush = True)

if device == torch.device("cuda"):
    print("GPU Name: ",torch.cuda.get_device_name(0), flush = True)
    
print("-------------------------------", flush = True)

working_dir = os.getcwd()
os.makedirs(working_dir, exist_ok=True)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Reading Data")
# Load preprocessed training data (macrostates) from a pickle file.
onne_data = pickle.load(open(onne_file, "rb"))

# Add if statement if needs filtering (if np.all(datapoint[1][:10] == 0)
onne_data = np.array([datapoint for datapoint in onne_data], dtype = object)

# Count total number of training samples
num_of_datapoints = len(onne_data)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Data size after filtering: {num_of_datapoints}")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Generating Validation Data")

#%%
if val == "onne":
    x_val, y_val = next(data_generator(batch_size = 50000, num_of_data_samples = num_of_datapoints, data = onne_data))
else:
    val_data = pickle.load(open(val,"rb"))
    # Data filter if needed 
    val_data = np.array([datapoint for datapoint in val_data if np.all(datapoint[1][:10] == 0)], dtype = object)
    x_val, y_val = np.vstack(val_data[:,0]), np.vstack(val_data[:,1])

x_val, y_val = torch.from_numpy(x_val).to(torch.float32), torch.from_numpy(y_val).to(torch.float32)

#  Wrap validation tensors into a PyTorch `TensorDataset` for batching
validation_dataset = TensorDataset(x_val, y_val)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Done")
print("-------------------------------", flush = True)

output_layer = len(y_val[0]) # Output layer is equal to the size of features

#%%
print("-------------------------------", flush = True)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Starting Random Search", flush = True)
print("-------------------------------", flush = True)

SEED = 1202; rng=np.random.default_rng(SEED)
# Constants
N_trials = 50
epochs = 10
# No. of batches per epoch for "data_generator"
records=[]; best=(10,None,None)

for n in range(N_trials):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Trial {n+1}", flush =True)
    print("-------------------------------", flush = True)
    
    # Batch size
    batch_size = int(2**rng.integers(4,8))
    batches_per_epoch = int(rng.integers(500,1000, endpoint = True))
                            
    train_generator = data_generator(batch_size = batch_size, 
                                    num_of_data_samples = num_of_datapoints,
                                    data = onne_data)
    
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Learning rate
    lr = float(10**rng.uniform(-4.1,-3.1))
    weight_decay = float(10**rng.uniform(-6.1,-3.9))

    
    # Dropout
    p = rng.choice([0.2,0.3,0.4,0.5], size = 1, replace = False).item()

    print(f"trial: {n+1}, LR: {lr: .7f}, Batch Size: {batch_size}, Iterations: {batches_per_epoch}, Weight Decay: {weight_decay: .7f}, Dropout:{p: .7f}", flush = True)
    model = CNNModel(output_layer, p)
    model, history= train_model(
                model=model,
                epochs=epochs,
                learning_rate=lr,
                train_generator=train_generator,
                val_loader=val_dataloader,
                device=device,
                steps_per_epoch=batches_per_epoch,
                weight_decay=weight_decay,
                early_stop = False
    )

    val_mse = history["val_loss"][-1]
    records.append((n+1,batch_size,batches_per_epoch,lr,weight_decay,p,val_mse))
    print(f"Val MSE:{val_mse: .7f}", flush = True)
    print("---------------------------------------------------------", flush = True)
    if val_mse < best[0]: best=(val_mse, {"lr":lr,"batch_size":batch_size,"iterations":batches_per_epoch,"weight_decay":weight_decay,"Dropout":p})

print(f"\nBest:{best}",flush = True)

record_file = os.path.join(working_dir, "record.txt")

with open(record_file, "w") as f:
    # Write header
    f.write('trial, batch_size, Iterations,lr, weight_decay, Dropout ,Val_MSE \n')

    # Write each tuple as a comma-separated line
    for r in records:
        line = ",".join(map(str, r))  # Convert tuple elements to strings
        f.write(line + "\n")
    f.close()
    
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Done with Hyperparameter tuning")
print("---------------------------------------------------------", flush = True)