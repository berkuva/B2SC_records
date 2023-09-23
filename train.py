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



label_map = {v:k for k,v in paired_dataset.mapping_dict.items()}

color_map = {
    'CD8+ NKT-like cells':'pink',
    'Classical Monocytes':'orange',
    'Effector CD4+ T cells':'grey',
    'Macrophages':'tan',
    'Myeloid Dendritic cells':'green',
    'Naive B cells':'red',
    'Naive CD4+ T cells':'slateblue',
    'Naive CD8+ T cells':'blue',
    'Natural killer  cells':'cyan',
    'Non-classical monocytes':'black',
    'Plasma B cells': 'purple', 
    'Plasmacytoid Dendritic cells':'lime',
    'Pre-B cells':'cornflowerblue',
    
}



def remove_params_from_optimizer(optimizer, params_to_remove):
    for param in params_to_remove:
        for group in optimizer.param_groups:
            # Use the id() function to check the identity of tensors
            group['params'] = [p for p in group['params'] if id(p) != id(param)]
    return optimizer


def to_float(gmm_weights_backup):
    float_dict = {name: tensor.float() for name, tensor in gmm_weights_backup.items()}
    return float_dict

train_gmm_till = 500

# Training Loop with warm-up
def train(epoch, model, optimizer, train_loader, gmm_weights_backup):
    print(f'Epoch: {epoch+1}')
    model.train()
    train_loss = 0
    
    for batch_idx, (data,labels) in enumerate(train_loader):
        data = data.to(torch.float32)

        data = data.to(device)
        # model = nn.DataParallel(model).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        preds, zero_inflation_prob, theta, mus, logvars, gmm_weights = model(data)
        # preds.shape, zero_inflation_prob.shape, theta.shape, mus[0].shape, logvars[0].shape, gmm_weights.shape
        # (torch.Size([mb, feat]), torch.Size([mb, feat]), torch.Size([feat]), torch.Size([mb, z_dim]), torch.Size([5000, 12]), torch.Size([5000, 12]))
        if train_gmm_till > epoch+1:
            loss = 10*losses.gmm_loss(gmm_weights)
        else:
            # recon_L = losses.zinb_loss(preds, data, zero_inflation_prob, theta)
            ceL = nn.CrossEntropyLoss()(preds, data)
            loss = ceL
            
        loss = loss.to(torch.float32)

        if train_gmm_till == epoch+1:
            gmm_weights_backup = {name: param.clone() for name, param in model.fc_gmm_weights.named_parameters()}
            gmm_weights_backup = to_float(gmm_weights_backup)

            for param in model.fc_gmm_weights.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        
        if epoch != 0:
            loss.backward()
            optimizer.step()

        if epoch+1 > train_gmm_till:
            for name, param in model.fc_gmm_weights.named_parameters():
                param.data = gmm_weights_backup[name]

        if epoch+1 == 0:
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
                # import pdb;pdb.set_trace()
                label_names = np.array([label_map[label] for label in raw_labels])
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
        import sys;sys.exit()
    
    if (epoch+1)%100 == 0:
        print(f'====> Epoch: {epoch+1} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    if (epoch+1) < train_gmm_till:
        print(f'GMM loss: {loss.item():.4f}')
    else:
        print(f'BCE loss: {ceL.item():.4f}')

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
            # import pdb;pdb.set_trace()
            label_names = np.array([label_map[label] for label in raw_labels])
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


# model.load_state_dict(torch.load("sc_model_600.pt"))
# gmm_weights_backup = {name: param.clone() for name, param in model.fc_gmm_weights.named_parameters()}

# # Set requires_grad of those parameters to False and zero their gradients
# params_to_remove_from_optim = []
# for param in model.fc_gmm_weights.parameters():
#     param.requires_grad = False
#     if param.grad is not None:
#         param.grad.data.zero_()
#     params_to_remove_from_optim.append(param)

# # Remove the parameters from the optimizer
# optimizer = remove_params_from_optimizer(optimizer, params_to_remove_from_optim)

# # Reset the optimizer learning rate
# for group in optimizer.param_groups:
#     group['lr'] = 1e-4

epoch_start = -1

print(paired_dataset.X_tensor.shape)
gmm_weights_backup = None

for epoch in range(epoch_start, epochs + 1):
    gmm_weights_backup = train(epoch, model, optimizer, train_loader, gmm_weights_backup)
    if gmm_weights_backup:
        gmm_weights_backup = to_float(gmm_weights_backup)



