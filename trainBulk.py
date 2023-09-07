import models
import losses
import data
from data import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

# Define hyperparameters and other settings
input_dim = 32738
hidden_dim = 700
z_dim = data.z_dim
epochs = 1000

training_losses = []
def train(epoch, scmodel, bulkmodel, optimizer, train_loader, scheduler):
    # print(epoch)
    bulkmodel.train()
    bulkmodel.to(device)
    
    train_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):


        adata_input = torch.FloatTensor(adata.X).to(device)
        sc_mus, sc_logvars, sc_gmm_weights = scmodel.encode(adata_input)

        sc_gmm_weights = torch.cat([sc_gmm_weights.mean(dim=0).reshape(1,-1),sc_gmm_weights.mean(dim=0).reshape(1,-1)],0)
        sc_mus_tensor = torch.stack(sc_mus)
        sc_logvars_tensor = torch.stack(sc_logvars)
        concatenated_sc_mus1 = torch.cat([sc_mus_tensor[:, :500, :].mean(1, keepdim=True), sc_mus_tensor[:, 500:1000, :].mean(1, keepdim=True)], dim=1)
        concatenated_sc_logvars1 = torch.cat([sc_logvars_tensor[:, :500, :].mean(1, keepdim=True), sc_logvars_tensor[:, 500:1000, :].mean(1, keepdim=True)], dim=1)
        concatenated_sc_mus2 = torch.cat([sc_mus_tensor[:, 1000:1500, :].mean(1, keepdim=True), sc_mus_tensor[:, 1500:2000, :].mean(1, keepdim=True)], dim=1)
        concatenated_sc_logvars2 = torch.cat([sc_logvars_tensor[:, 1000:1500, :].mean(1, keepdim=True), sc_logvars_tensor[:, 1500:2000, :].mean(1, keepdim=True)], dim=1)
        concatenated_sc_mus3 = torch.cat([sc_mus_tensor[:, 2000:2500, :].mean(1, keepdim=True), sc_mus_tensor[:, 2500:2700, :].mean(1, keepdim=True)], dim=1)
        concatenated_sc_logvars3 = torch.cat([sc_logvars_tensor[:, 2000:2500, :].mean(1, keepdim=True), sc_logvars_tensor[:, 2500:2700, :].mean(1, keepdim=True)], dim=1)


        if batch_idx == 0:
            concatenated_sc_mus = concatenated_sc_mus1.clone()
            concatenated_sc_logvars = concatenated_sc_logvars1.clone()
        elif batch_idx == 1:
            concatenated_sc_mus = concatenated_sc_mus2.clone()
            concatenated_sc_logvars = concatenated_sc_logvars2.clone()
        elif batch_idx == 2:
            concatenated_sc_mus = concatenated_sc_mus3.clone()
            concatenated_sc_logvars = concatenated_sc_logvars3.clone()
        sc_gmm_weights = sc_gmm_weights.clone()

    
        # Shuffle the rows of the data tensor
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]

        # Split the shuffled data into two halves
        mid_idx = shuffled_data.size(0) // 2
        first_half, second_half = shuffled_data[:mid_idx], shuffled_data[mid_idx:]

        # Sum the rows for each half
        data1 = torch.sum(first_half, dim=0).reshape(-1, input_dim)
        data2 = torch.sum(second_half, dim=0).reshape(-1, input_dim)
        # concatenate the two halves
        data = torch.cat((data1, data2), dim=0)

        data = data.to(device)

        
        bulk_mus, bulk_logvars, bulk_gmm_weights = bulkmodel(data)
        bulk_mus = torch.stack(bulk_mus)
        bulk_logvars = torch.stack(bulk_logvars)

        mus_loss = nn.MSELoss()(concatenated_sc_mus, bulk_mus)
        logvars_loss = nn.MSELoss()(concatenated_sc_logvars, bulk_logvars)

        gmm_weights_loss = nn.MSELoss()(bulk_gmm_weights, sc_gmm_weights) + nn.L1Loss()(bulk_gmm_weights, sc_gmm_weights)

        # BulkLoss = losses.bulk_loss(current_sc_mus, current_sc_logvars, bulk_mus, bulk_logvars)
        
        combined_loss = mus_loss + logvars_loss + gmm_weights_loss
        # if batch_idx == 0:
        #     combined_loss = logvars_loss + gmm_weights_loss
        # else:
        #     combined_loss = mus_loss + logvars_loss+ gmm_weights_loss
        # pdb.set_trace()
        combined_loss.backward()

        optimizer.step()
        optimizer.zero_grad() 
        scheduler.step()
        
        train_loss += combined_loss.item()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    print(f'Mus loss: {mus_loss.item():.4f}, Logvars loss: {logvars_loss.item():.4f}, GMM weights loss: {gmm_weights_loss.item():.4f}')
    if (epoch+1)%50 == 0:
        print(bulk_gmm_weights.mean(0))
    if (epoch+1)%100 == 0:
        print(sc_gmm_weights.mean(0))

    if (epoch+1)%500 == 0:
        bulkmodel = bulkmodel.eval()
        torch.save(bulkmodel.cpu().state_dict(), f"bulk_model_{epoch+1}.pt")
    

if __name__ == "__main__":
    train_loader = data.loader
    bulk_model = models.bulkVAE(input_dim, hidden_dim, z_dim)
    optimizer = torch.optim.Adam(bulk_model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    scmodel = models.scVAE(input_dim, hidden_dim, z_dim)

    # Load the state dictionary and modify the keys
    state_dict = torch.load('model_1000.pt')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    scmodel.load_state_dict(new_state_dict)
    scmodel = scmodel.to(device)


    for param in scmodel.parameters():
        param.requires_grad = False

    bulk_model = bulk_model.to(device)

    print("Loaded model.")

    for epoch in range(1, epochs + 1):

        
        train(epoch,\
              scmodel,
               bulk_model,
               optimizer,
               train_loader,
               scheduler)

    # After all your training epochs are complete
    plt.figure(figsize=(10,5))
    plt.plot(training_losses)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Bulk_training_loss.png')