import torch
import numpy as np
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F

def training_step(model, batch, device):
    images, labels = batch 
    images, labels = images.to(device), labels.to(device)
    out, *_ = model(images)                  # Generate predictions
    loss = F.l1_loss(out, labels) # Calculate loss
    return loss

def validation_step(model, batch, device):
    images, labels= batch 
    images, labels = images.to(device), labels.to(device)
    out, *_ = model(images)                    # Generate predictions
    loss = F.l1_loss(out, labels)   # Calculate loss
    return {'Loss': loss.detach()}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    return {'Loss': epoch_loss.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['Loss']))



@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs,  model, train_loader, val_loader, device, save_path, lr=0.01):
    best_loss = np.inf
    torch.cuda.empty_cache()
    history = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        
        # Validation phase
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        sched.step(result['Loss'])
        if best_loss > result['Loss']:
            best_loss = result['Loss']
            torch.save(model.state_dict(), save_path)
    
    return history

def inference_step(model, batch, device):
    images, labels= batch 
    images, labels = images.to(device), labels.to(device)
    out, *_ = model(images)                    # Generate predictions
    return out


@torch.no_grad()
def predict(model, val_loader, device):
    model.eval()
    outputs = [inference_step(model, batch, device) for batch in val_loader]
    return torch.cat(outputs, axis = 0)