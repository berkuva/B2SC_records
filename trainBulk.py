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

        mu1, logvar1, mu2, logvar2, mu3, logvar3, mu4, logvar4, mu5, logvar5, mu6, logvar6, mu7, logvar7, mu8, logvar8, mu9, logvar9 = scmodel.encode(data)

        bulkmodel = bulkmodel.train()
        optimizer.zero_grad()

        mus, logvars = bulkmodel(data)
        bulkmu1, bulkmu2, bulkmu3, bulkmu4, bulkmu5, bulkmu6, bulkmu7, bulkmu8, bulkmu9 = mus
        bulklogvar1, bulklogvar2, bulklogvar3, bulklogvar4, bulklogvar5, bulklogvar6, bulklogvar7, bulklogvar8, bulklogvar9 = logvars

        BulkLoss = losses.bulk_loss(mu1, logvar1, mu2, logvar2, mu3, logvar3, mu4, logvar4, mu5, logvar5,\
                                    mu6, logvar6, mu7, logvar7, mu8, logvar8, mu9, logvar9,\
                                    bulkmu1, bulklogvar1, bulkmu2, bulklogvar2, bulkmu3, bulklogvar3,\
                                    bulkmu4, bulklogvar4, bulkmu5, bulklogvar5, bulkmu6, bulklogvar6,\
                                    bulkmu7, bulklogvar7, bulkmu8, bulklogvar8, bulkmu9, bulklogvar9)
        
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
    scmodel.load_state_dict(torch.load('Pmodel_1000.pt'))
    
    for layer_name in ['fc1', 'fc1_mean', 'fc1_logvar', 'fc2', 'fc2_mean', 'fc2_logvar',
                   'fc3', 'fc3_mean', 'fc3_logvar', 'fc4', 'fc4_mean', 'fc4_logvar',
                   'fc5', 'fc5_mean', 'fc5_logvar', 'fc6', 'fc6_mean', 'fc6_logvar',
                   'fc7', 'fc7_mean', 'fc7_logvar', 'fc8', 'fc8_mean', 'fc8_logvar',
                   'fc9', 'fc9_mean', 'fc9_logvar']:
        getattr(bulk_model, layer_name).load_state_dict(getattr(scmodel, layer_name).state_dict())

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