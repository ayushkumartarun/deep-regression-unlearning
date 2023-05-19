import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from utils import get_lr, evaluate, epoch_end
    
class UAgeDB(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train'):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')
        ulabel = np.asarray(row['unlearn']).astype('float32')

        return img, label, ulabel

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

def attention(x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()



def forget_loss(model_output, model_activations, proxy_output, proxy_activations, mask, AT_beta = 50):

    loss = F.l1_loss(model_output[mask], proxy_output[mask])
    if AT_beta > 0:
        at_loss = 0
        for i in range(len(proxy_activations)):
            at_loss = at_loss + AT_beta * attention_diff(model_activations[i][mask], proxy_activations[i][mask])
    else:
        at_loss = 0

    total_loss = loss + at_loss

    return total_loss



def fit_one_forget_cycle(epochs,  model, proxy_model, train_loader, val_loader, lr, device, save_path):
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
            
            images, labels, ulabels = batch 
            images, labels, ulabels = images.to(device), labels.to(device), ulabels.to(device)
            model_out, *model_activations = model(images)
            with torch.no_grad():
                proxy_out, *proxy_activations = proxy_model(images)
                
            
            label_loss = 0
            if ulabels.sum() < len(ulabels):
                mask = (ulabels == 0)
                r_model_out = model_out[mask]
                r_labels = labels[mask]
                label_loss = F.l1_loss(r_model_out, r_labels)
            
            proxy_loss = 0
            if ulabels.sum() > 0:
                mask = (ulabels == 1)
                proxy_loss = forget_loss(model_out, model_activations, proxy_out, proxy_activations, mask)
            
            coeff = ulabels.sum()/len(ulabels)
            loss = coeff*proxy_loss + (1-coeff)*label_loss
            
            ######
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
        torch.save(model.state_dict(), save_path)
    
    return history