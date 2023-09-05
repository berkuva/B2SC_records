import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import models
import data
from data import *
import losses
import torch.nn as nn
import umap
import pdb


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

# Define hyperparameters and other settings
input_dim = 32738
hidden_dim = 700
z_dim = 9
epochs = 1001


# Create VAE and optimizer
model = models.scVAE(input_dim, hidden_dim, z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# Weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight) 
        m.bias.data.fill_(0.01)

model.apply(init_weights) 
model = model.to(device)
model = nn.DataParallel(model).to(device)

# Clamp weights
for m in model.modules():
    if isinstance(m, nn.Linear):
        m.weight.data.clamp_(-5, 5)

train_loader = data.loader 


# Training Loop with warm-up
def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    
    for batch_idx, (data,labels) in enumerate(train_loader):
        data = data.to(device)
        # model = nn.DataParallel(model).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        gmm_weights = model.module.gmm_weights


        if epoch+1 < 201:
            gmm_loss = 10*losses.gmm_loss(gmm_weights)
            loss = gmm_loss
       
        elif epoch+1 < 600:
            recon_batch, mus, logvars = model(data)
            total_count, probs, logits = recon_batch
            prob_loss = F.binary_cross_entropy(probs, data, reduction='sum')
            loss = prob_loss
        
        elif epoch+1 < 800:
            recon_batch, mus, logvars = model(data)
            total_count, probs, logits = recon_batch
            prob_loss = F.binary_cross_entropy(probs, data, reduction='sum')
            kld = losses.KLDiv(mus, logvars, model.module.gmm_weights)
            loss = prob_loss+kld.clamp(-10000, 10000)
        else:
            recon_batch, mus, logvars = model(data)
            total_count, probs, logits = recon_batch
            prob_loss = F.binary_cross_entropy(probs, data, reduction='sum')
            kld = losses.KLDiv(mus, logvars, model.module.gmm_weights)

            # dist_loss = models.PoissonDist(total_count.clamp(0, torch.max(data).item()))
            dist_loss = models.ZIP(total_count.clamp(0, torch.max(data).item()),logits)
            # dist_loss = models.ZINB(total_count.clamp(0, torch.max(data).item()), probs, logits)
            recon = dist_loss.log_prob(data.view(-1, input_dim)).sum()
            
            loss = prob_loss+kld.clamp(-10000, 10000)+recon.clamp(-10000, 10000)

        loss.backward()


        if epoch == 200:
            model.module.gmm_weights.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

            
        train_loss += loss.item()
        optimizer.step()


    if (epoch+1)%10 == 0:
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    
    if (epoch+1)%500 == 0:
            # save model checkpoint
            torch.save(model.cpu().state_dict(), f"model_{epoch+1}.pt")

    if (epoch+1)%100 == 0:
        model.eval()
        model.to(device)
        with torch.no_grad():
            recon_batch, mus, logvars = model(data)
            
            # mus, logvars = model.encode(torch.Tensor(adata.X).to(device))
            mus, logvars = model.module.encode(torch.Tensor(adata.X).to(device))

            # z = model.reparameterize(mus, logvars).cpu().numpy()
            z = model.module.reparameterize(mus, logvars).cpu().numpy()


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

            # Create a list that converts adata.obs['labels'] to their respective names
            celltype_labels = [label_map[l] for l in adata.obs['labels']]

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

            reducer = TSNE()

            # Joint embedding
            embedding = reducer.fit_transform(z)
            plt.figure(1, figsize=(8, 8))
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=celltype_labels, legend='full', palette=color_map)
            plt.title('t-SNE for joint representation')
            plt.savefig(f'tsne_joint_{epoch+1}.png')
            plt.close()


# Load model_200.pt
# model.load_state_dict(torch.load("model_1000.pt"))
# pdb.set_trace()
for epoch in range(1, epochs + 1):
    train(epoch, model, optimizer, train_loader)
    if (epoch+1)%100 == 0:
        print(model.module.gmm_weights)

    # if epoch >200:
    #     if model.module.gmm_weights.requires_grad:
    #         model.module.gmm_weights.grad.zero_()
    
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
