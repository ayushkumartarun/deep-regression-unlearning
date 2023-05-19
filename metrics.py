import torch
import numpy as np
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tqdm
import random
from sklearn.svm import SVC
import numpy as np
from torch.utils import data
from utils import evaluate, training_step

def get_attack_features(data_loader, model, device='cuda'):
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False)#, num_workers = 8, prefetch_factor = 4)
    prefinal_gradients = []
    prefinal_activations = []
    predictions = []
    labels = []
    losses = []

    model.eval()
    for batch in data_loader:
        data, target = batch
        data, target = data.to(device), target.to(device)
        output, *_ = model(data)
        
        loss = F.l1_loss(output, target)

        
        predictions.append(output.detach().cpu())
        labels.append(target.detach().cpu())
        losses.append(loss.detach().cpu())

    predictions = torch.squeeze(torch.cat(predictions, axis = 0))
    labels = torch.squeeze(torch.cat(labels, axis = 0))
    losses = torch.tensor(losses)
    
    return predictions, labels, losses

def get_membership_attack_data(retain_loader, test_loader, model, prediction_loaders = None):    
    retain_preds, retain_labels, retain_losses = get_attack_features(retain_loader, model)    
    test_preds, test_labels, test_losses = get_attack_features(test_loader, model)
    
    prediction_outputs = {}
    if prediction_loaders is not None:
        for prediction_set, prediction_loader in prediction_loaders.items():
            prediction_preds, prediction_labels, prediction_losses = get_attack_features(prediction_loader, model)

            prediction_outputs[prediction_set] = {'prediction_preds':prediction_preds, 
                                                  'prediction_labels':prediction_labels, 
                                                  'prediction_losses':prediction_losses}
    
    prediction_outputs['train'] = {'prediction_preds':retain_preds, 
                                      'prediction_labels':retain_labels, 
                                      'prediction_losses':retain_losses}
    
    prediction_outputs['test'] = {'prediction_preds':test_preds, 
                                      'prediction_labels':test_labels, 
                                      'prediction_losses':test_losses}
    
    X_train = torch.cat([retain_losses, test_losses], axis = 0).numpy()
    Y_train = np.concatenate([np.ones(len(retain_losses)), np.zeros(len(test_losses))])
    
    index_shuf = list(range(len(X_train)))
    random.Random().shuffle(index_shuf)
    X_train = X_train[index_shuf]
    Y_train = Y_train[index_shuf]
    
    return X_train, Y_train, prediction_outputs

def get_membership_attack_prob(retain_loader, test_loader, model, prediction_loaders = None):
    X_train, Y_train, prediction_outputs = get_membership_attack_data(retain_loader, test_loader, model, prediction_loaders)
    clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf.fit(X_train[:, np.newaxis], Y_train[:, np.newaxis])
    
    results = {}
    for prediction_set, features in prediction_outputs.items():
        attack_result = clf.predict(features['prediction_losses'][:, np.newaxis])
        results[prediction_set] = attack_result.mean()
    return results

def relearn_time(model, train_loader, valid_loader, reqAcc, lr, device = 'cuda'):
    rltime = 0
    curr_Acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(10):
        
        for batch in train_loader:
            model.train()
            loss = training_step(model, batch, device)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            history = [evaluate(model, valid_loader, device)]
            curr_Acc = history[0]["Loss"]
            
            rltime += 1
            if(curr_Acc <= reqAcc):
                break
                
        if(curr_Acc <= reqAcc):
            break
    return rltime

def ain(full_model, model, gold_model, train_data, val_retain, val_forget, 
                  batch_size = 256, error_range = 0.05, lr = 0.001, device = 'cuda'):
    # measuring performance of fully trained model on forget class
    forget_valid_dl = DataLoader(val_forget, batch_size, num_workers = 64)
    history = [evaluate(full_model, forget_valid_dl, device)]
    LossForget = history[0]["Loss"]
    
    retain_valid_dl = DataLoader(val_retain, batch_size, num_workers = 64)
    history = [evaluate(full_model, retain_valid_dl, device)]
    LossRetain = history[0]["Loss"]
    
    history = [evaluate(model, forget_valid_dl, device)]
    LossForget_Fmodel = history[0]["Loss"]
    
    history = [evaluate(model, retain_valid_dl, device)]
    LossRetain_Fmodel = history[0]["Loss"]
    
    history = [evaluate(gold_model, forget_valid_dl, device)]
    LossForget_Gmodel = history[0]["Loss"]
    
    history = [evaluate(gold_model, retain_valid_dl, device)]
    LossRetain_Gmodel = history[0]["Loss"]
    
    reqLossF = (1+error_range)*LossForget
    
    train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers = 64)
    valid_loader = DataLoader(val_forget, batch_size, num_workers = 64)
    rltime_gold = relearn_time(model = gold_model, train_loader = train_loader, valid_loader = valid_loader, 
                               reqAcc = reqLossF,  lr = lr, device = device)
    
    rltime_forget = relearn_time(model = model, train_loader = train_loader, valid_loader = valid_loader, 
                               reqAcc = reqLossF, lr = lr, device = device)
    
    rl_coeff = rltime_forget/rltime_gold
    return rl_coeff