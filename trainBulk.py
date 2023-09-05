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
z_dim = 9
epochs = 401



training_losses = []
def train(epoch, bulkmodel, optimizer, train_loader):
    bulkmodel.train()
    bulkmodel.to(device)
    
    train_loss = 0
    
    for batch_idx, (data,_) in enumerate(train_loader):

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

        sc_mus, sc_logvars = scmodel.encode(data)
        bulkmodel = bulkmodel.train()
        optimizer.zero_grad()

        bulk_mus, bulk_logvars = bulkmodel(data)

        BulkLoss = losses.bulk_loss(sc_mus, sc_logvars, bulk_mus, bulk_logvars)
        
        loss = BulkLoss

        loss.backward()
        train_loss += loss.item()
        
        optimizer.step()



    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    training_losses.append(BulkLoss.cpu().item())


    if (epoch+1)%200 == 0:
        bulkmodel = bulkmodel.eval()
        torch.save(bulkmodel.cpu().state_dict(), f"Pbulk_model_{epoch+1}.pt")
    
        


if __name__ == "__main__":

    train_loader = data.loader
    bulk_model = models.bulkVAE(input_dim, hidden_dim, z_dim)
    optimizer = torch.optim.Adam(bulk_model.parameters(), lr=1e-3)
    scmodel = models.scVAE(input_dim, hidden_dim, z_dim)

    # Load the state dictionary and modify the keys
    state_dict = torch.load('model_1000.pt')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    scmodel.load_state_dict(new_state_dict)
    # Transfer encoder weights from layers 1 to 9.
    sc_encoder_layers = ['fcs', 'fcs_mean', 'fcs_logvar']
    bulk_encoder_layers = ['fcs', 'fc_means', 'fc_logvars']

    for idx in range(bulk_model.num_gmms):
        for sc_layer, bulk_layer in zip(sc_encoder_layers, bulk_encoder_layers):
            getattr(bulk_model, bulk_layer)[idx].load_state_dict(
                getattr(scmodel, sc_layer)[idx].state_dict())

    scmodel = scmodel.to(device)

    bulk_model.gmm_weights = scmodel.gmm_weights
    bulk_model = bulk_model.to(device)

    print("Loaded model.")

    for epoch in range(1, epochs + 1):
        train(epoch, bulk_model, optimizer, train_loader)
    # After all your training epochs are complete
    plt.figure(figsize=(10,5))
    plt.plot(training_losses)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Bulk_training_loss.png')