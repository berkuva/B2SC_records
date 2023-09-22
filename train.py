import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import models
import paired_dataset
from paired_dataset import *
import losses
import torch.nn as nn
import umap
import pdb
import time

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

# Define hyperparameters and other settings
input_dim = paired_dataset.input_dim
hidden_dim = paired_dataset.hidden_dim
epochs = 1200
z_dim = paired_dataset.z_dim


# Create VAE and optimizer
model = models.scVAE(input_dim, hidden_dim, z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# Weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight) 
        m.bias.data.fill_(0.01)

model.apply(init_weights) 
model = model.to(device)

# Clamp weights
for m in model.modules():
    if isinstance(m, nn.Linear):
        m.weight.data.clamp_(-5, 5)

train_loader = paired_dataset.dataloader 


label_map = {
    '0': 'Naive B cells',
    '1': 'Non-classical monocytes',
    '2': 'Classical Monocytes',
    '3': 'Natural killer cells',
    '4': 'Naive CD8+ T cells',
    '5': 'Memory CD4+ T cells',
    '6': 'CD8+ NKT-like cells',
    '7': 'Myeloid Dendritic cells',
    '8': 'Platelets',
}


color_map = {
    'Naive B cells': 'red',
    'Classical Monocytes': 'orange',
    'Platelets': 'yellow',
    'Myeloid Dendritic cells': 'green',
    'Naive CD8+ T cells': 'blue',
    'Non-classical monocytes': 'black',
    'Memory CD4+ T cells': 'purple',
    'CD8+ NKT-like cells': 'pink',
    'Natural killer cells': 'cyan'  # Fixed the typo here
}

train_gmm_till = 200

# Training Loop with warm-up
def train(epoch, model, optimizer, train_loader, gmm_weights_backup):
    print(f'Epoch: {epoch+1}')
    model.train()

        
    train_loss = 0
    
    for batch_idx, (data,labels) in enumerate(train_loader):
        # Check how much time it takes for each iteration.
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        recon_batch, mus, logvars, gmm_weights = model(data)
        total_count, probs, logits = recon_batch

        if epoch+1 < train_gmm_till:
            gmm_loss = 10*losses.gmm_loss(gmm_weights)
            loss = gmm_loss
       
        elif epoch+1 < 500:
            total_count, probs, logits = recon_batch
            prob_loss = F.binary_cross_entropy(probs, data, reduction='sum')
            loss = prob_loss
        
        elif epoch+1 < 700:
            total_count, probs, logits = recon_batch
            kld = losses.KLDiv(mus, logvars, gmm_weights)
            loss = kld.clamp(-10000, 10000)

        else:
            total_count, probs, logits = recon_batch
            prob_loss = F.binary_cross_entropy(probs, data, reduction='sum')

            loss = prob_loss


        train_loss += loss.item()
        
        if epoch+1 == train_gmm_till:
            gmm_weights_backup = {name: param.clone() for name, param in model.fc_gmm_weights.named_parameters()}
            for param in model.fc_gmm_weights.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        loss.backward()
        optimizer.step()

        if epoch+1 >= train_gmm_till:
            for name, param in model.fc_gmm_weights.named_parameters():
                param.data = gmm_weights_backup[name]

    if (epoch+1)%10 == 0:
        print(f'Average loss: {train_loss / len(train_loader.dataset):.4f}')

    if (epoch+1)%20 == 0:
        print(gmm_weights)

    if (epoch+1)%100 == 0:
        torch.save(model.cpu().state_dict(), f"sc_model_{epoch+1}.pt")
        X_tensor = paired_dataset.X_tensor
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            # recon_batch, mus, logvars, gmm_weights = model(X_tensor)
            mus, logvars, gmm_weights = model.encode(X_tensor.to(device))
            z = model.reparameterize(mus, logvars, gmm_weights).cpu().numpy()

            # Convert cell_types_tensor to numpy for plotting
            raw_labels = paired_dataset.cell_types_tensor.numpy()

            # Convert raw_labels to their names using the label_map
            label_names = np.array([label_map[str(label)] for label in raw_labels])
            unique_labels = np.unique(label_names)

            # embedding
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(z)

            # Plot the UMAP representation
            plt.figure(figsize=(10, 10))
            for label in unique_labels:
                indices = np.where(label_names == label)
                plt.scatter(embedding[indices, 0], embedding[indices, 1], color=color_map[label], label=label)
            plt.legend()
            plt.savefig(f"umap_{epoch+1}.png")


    return gmm_weights_backup
            
epoch_start = 0
print(paired_dataset.X_tensor.shape)
gmm_weights_backup = None
for epoch in range(epoch_start, epochs + 1):

    gmm_weights_backup = train(epoch, model, optimizer, train_loader, gmm_weights_backup)

