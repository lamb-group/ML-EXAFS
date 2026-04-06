import torch, torch.nn as nn, torch.optim as optim

from datetime import datetime
from earlystopping import EarlyStopping

def save_checkpoint(epoch, model, optimizer, loss, path):
    """
    Saves the training checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    ckpt_file = f"{path}_{epoch}.pth"
    torch.save(checkpoint, ckpt_file)
    print(f"Checkpoint saved to {ckpt_file}")

def train_model(model, epochs, learning_rate, train_generator, val_loader, device, steps_per_epoch, weight_decay, path = None, early_stop = False):
    """
    Train a PyTorch model using data from a generator.

    Args:
        model: The PyTorch model to train.
        epochs: Number of epochs to train for.
        learning_rate: Learning rate for the optimizer.
        train_generator: A generator function that yields (inputs, targets) for training.
        val_loader: DataLoader for validation data.
        device: The device (CPU or GPU) to train on.
        steps_per_epoch: Number of batches per epoch.
        batch_size: Batch size used in training.
    """
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-6)


    if early_stop:
        earlystopping = EarlyStopping(patience = 5, delta = 1e-5, verbose = False)
    else:
        earlystopping = EarlyStopping(patience = epochs, delta = 0, verbose = False)
        
    model.to(device)

    # Initialize lists to store losses
    train_losses, val_losses, lr = [],[],[]

    # Loop through epochs
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        # Iterate over training data
        for _ in range(steps_per_epoch):

            # Get a batch from the generator
            x_batch, y_batch = next(train_generator)
            
             # Convert to PyTorch tensors and send to device
            x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_train_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = total_train_loss / steps_per_epoch
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                #batch_size, length = x_val.shape
                x_val, y_val = x_val.to(device), y_val.to(device) #x_val.reshape(batch_size,1,length)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                total_val_loss += loss.item()

        avg_val_loss= total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_lr = optimizer.param_groups[0]["lr"]
        lr.append(epoch_lr)
        
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.7f}, "
              f"Validation Loss: {avg_val_loss:.7f}, "
              f"LR: {epoch_lr: .7f}", flush = True)
        
        # save_checkpoint(epoch, model, optimizer, loss, path)
        
        earlystopping(avg_val_loss)
        
        if earlystopping.early_stop:
            print(f"Early stopping at epoch {epoch}", flush = True)
            break
        
    history = {"train_loss": train_losses, "val_loss": val_losses,"lr":lr}
    return model, history


def fineTune_model(model, epochs, learning_rate, train_loader, val_loader, device, weight_decay, early_stop = False):
    """
    Train a PyTorch model using data from a generator.

    Args:
        model: The PyTorch model to train.
        epochs: Number of epochs to train for.
        learning_rate: Learning rate for the optimizer.
        train_generator: A generator function that yields (inputs, targets) for training.
        val_loader: DataLoader for validation data.
        device: The device (CPU or GPU) to train on.
        steps_per_epoch: Number of batches per epoch.
        batch_size: Batch size used in training.
    """


    # Fine tuninig

    for name, param in model.named_parameters():
        if "conv_block" in name:
            param.requires_grad = False
            
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,model.parameters()),
                            lr = learning_rate,
                            weight_decay = weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-6)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Frozen Layers:")
    print("---------------------------------------", flush = True)
    for name, param in model.named_parameters():
        print(f"{name}: ", param.requires_grad, flush = True)
    print("---------------------------------------", flush = True)

    if early_stop:
        earlystopping = EarlyStopping(patience = 5, delta = 1e-5, verbose = False)
    else:
        earlystopping = EarlyStopping(patience = epochs, delta = 0, verbose = False)
        
    model.to(device)

    # Initialize lists to store losses
    train_losses, val_losses, lr = [],[],[]

    # Loop through epochs
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        # Iterate over training data
        for x_train, y_train in train_loader:
            # Get a batch from the generator
            x_train, y_train = x_train.to(device), y_train.to(device)

            x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
            # Forward pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_train_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.reshape(x_val.shape[0],1,x_val.shape[1])
                x_val, y_val = x_val.to(device), y_val.to(device) #x_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                total_val_loss += loss.item()

        avg_val_loss= total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_lr = optimizer.param_groups[0]["lr"]
        lr.append(epoch_lr)
        
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.7f}, "
              f"Validation Loss: {avg_val_loss:.7f}, "
              f"LR: {epoch_lr: .7f}", flush = True)
        
        # save_checkpoint(epoch, model, optimizer, loss, path)
        
        earlystopping(avg_val_loss)
        
        if earlystopping.early_stop:
            print(f"Early stopping at epoch {epoch}", flush = True)
            break
        
    history = {"train_loss": train_losses, "val_loss": val_losses,"lr":lr}
    return model, history
    
def train_md_model(model, epochs, learning_rate, train_loader, val_loader, device, weight_decay):
    """
    Train a PyTorch model using data from a generator.

    Args:
        model: The PyTorch model to train.
        epochs: Number of epochs to train for.
        learning_rate: Learning rate for the optimizer.
        train_generator: A generator function that yields (inputs, targets) for training.
        val_loader: DataLoader for validation data.
        device: The device (CPU or GPU) to train on.
        steps_per_epoch: Number of batches per epoch.
        batch_size: Batch size used in training.
    """
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience = 5, delta = 1E-5, verbose = False)
    
    model.to(device)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    lr = []
    # Loop through epochs
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        # Iterate over training data
        for x_train, y_train in train_loader:
            # Get a batch from the generator
            x_train, y_train = x_train.to(device), y_train.to(device)

            x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
            # Forward pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_train_loss += loss.item()
        scheduler.step()

        # Calculate average training loss
        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)

                x_val = x_val.reshape(x_val.shape[0],1,x_val.shape[1])
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        epoch_lr = optimizer.param_groups[0]["lr"]
        lr.append(epoch_lr)
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {epoch_train_loss:.7f}, "
              f"Validation Loss: {avg_val_loss:.7f} "
              f"lr: {epoch_lr:.7f}")
        early_stopping(avg_val_loss)
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}", flush = True)
            break
            
    history = {"train_loss": train_losses, "val_loss": val_losses,"lr":lr}
    return model,history