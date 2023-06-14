import re
import torch
import numpy as np
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

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


def clean_text(text):
    # lower case characters only
    text = text.lower() 
    
    # remove urls
    text = re.sub('http\S+', ' ', text)
    
    # only alphabets, spaces and apostrophes 
    text = re.sub("[^a-z' ]+", ' ', text)
    
    # remove all apostrophes which are not used in word contractions
    text = ' ' + text + ' '
    text = re.sub("[^a-z]'|'[^a-z]", ' ', text)
    
    split_sentence = text.split()
    filtered_sentence = [w for w in split_sentence if not w.lower() in stop_words]
    return filtered_sentence

