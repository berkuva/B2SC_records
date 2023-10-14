import models
import losses
import paired_dataset
from paired_dataset import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

# Define hyperparameters and other settings
input_dim = paired_dataset.input_dim
hidden_dim = paired_dataset.hidden_dim
z_dim = paired_dataset.z_dim
epochs = 3000

training_losses = []
def train(epoch, bulkmodel, optimizer, train_loader, sc_mus, sc_logvars, sc_gmm_weights, gmm_weights_backup, train_gmm_till=1500):
    # print("Epoch: ", epoch+1)
    bulkmodel.train()
    bulkmodel.to(device)
    
    train_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        # adata_input = X_tensor.to(device)
        # indices = torch.randperm(data.size(0))#[:num_samples]  # Generate random indices
        # sampled_tensor = data[indices]

        # adata_input = sampled_tensor.to(device)
        
        # sc_mus, sc_logvars, sc_gmm_weights = scmodel.encode(data)

        # sc_gmm_weights = sc_gmm_weights.sum(dim=0).reshape(1,-1)
        sc_mus_tensor = torch.stack(sc_mus)
        sc_logvars_tensor = torch.stack(sc_logvars)
        # pdb.set_trace()
        
        concatenated_sc_mus = sc_mus_tensor.sum(1, keepdim=True)
        concatenated_sc_logvars = sc_logvars_tensor.sum(1, keepdim=True)

        data = torch.sum(data, dim=0).reshape(1,-1)

        data = data.to(device)
        
        bulk_mus, bulk_logvars, bulk_gmm_weights = bulkmodel(data)
        bulk_mus = torch.stack(bulk_mus)
        if torch.isnan(bulk_mus).any():
            pdb.set_trace()
        bulk_logvars = torch.stack(bulk_logvars)


        if epoch+1 <= train_gmm_till:
            gmm_weights_loss = nn.MSELoss()(bulk_gmm_weights.expand(len(sc_gmm_weights), paired_dataset.z_dim), sc_gmm_weights)
            combined_loss = gmm_weights_loss#max(train_gmm_till-epoch, 1)* 
        elif epoch+1 > train_gmm_till:
            gmm_weights_backup = {name: param.clone() for name, param in bulkmodel.fc_gmm_weights.named_parameters()}
            for param in bulkmodel.fc_gmm_weights.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, bulkmodel.parameters()), lr=1e-4)
        
            mus_loss = nn.MSELoss()(concatenated_sc_mus, bulk_mus.expand(paired_dataset.z_dim, concatenated_sc_mus.shape[1], paired_dataset.z_dim))
            logvars_loss = nn.MSELoss()(concatenated_sc_logvars, bulk_logvars.expand(paired_dataset.z_dim, concatenated_sc_logvars.shape[1], paired_dataset.z_dim))
            combined_loss = mus_loss + logvars_loss

        combined_loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        if epoch+1 > train_gmm_till:
            for name, param in bulkmodel.fc_gmm_weights.named_parameters():
                param.data = gmm_weights_backup[name]

        train_loss += combined_loss.item()

    if (epoch+1)%10 == 0:
        print(f'====> Epoch: {epoch+1} Average loss: {train_loss:.4f}')
        print(bulk_gmm_weights.mean(0))
        print(sc_gmm_weights.mean(0))
    elif (epoch+1)%10 == 0 and epoch+1 > train_gmm_till:
        print(f'====> Epoch: {epoch+1} Average loss: {train_loss:.4f}')
        print(mus_loss.item())
        print(logvars_loss.item())

    training_losses.append(combined_loss.item())
    
  
    if (epoch+1)%1000 == 0:
        bulkmodel = bulkmodel.eval()
        torch.save(bulkmodel.cpu().state_dict(), f"bulk_model_{epoch+1}.pt")
    
    return gmm_weights_backup
    
    
if __name__ == "__main__":
    

    train_loader = paired_dataset.dataloader
    bulk_model = models.bulkVAE(input_dim, hidden_dim, z_dim)
    optimizer = torch.optim.Adam(bulk_model.parameters(), lr=5e-3)


    scmodel = models.scVAE(input_dim, hidden_dim, z_dim)

    # Load the state dictionary and modify the keys
    state_dict = torch.load('sc_model_1500.pt')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    scmodel.load_state_dict(new_state_dict)
    scmodel = scmodel.to(device)

    for param in scmodel.parameters():
        param.requires_grad = False
        
    sc_mus, sc_logvars, sc_gmm_weights = scmodel.encode(paired_dataset.X_tensor.to(device))

    bulk_model = bulk_model.to(device)
    print("Loaded model.")
    gmm_weights_backup = None

    for epoch in range(0, epochs + 1):
        gmm_weights_backup = train(epoch,\
                                    bulk_model,
                                    optimizer,
                                    train_loader,
                                    sc_mus,
                                    sc_logvars,
                                    sc_gmm_weights,
                                    gmm_weights_backup)
        
