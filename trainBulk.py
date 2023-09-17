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
epochs = 1000

# def process_tensor(tensor, start_idx, end_idx):
#     return torch.cat([tensor[:, start_idx:start_idx+500, :].mean(1, keepdim=True), 
#                     tensor[:, start_idx+500:end_idx, :].mean(1, keepdim=True)], dim=1)


training_losses = []
def train(epoch, scmodel, bulkmodel, optimizer, train_loader, scheduler):
    print("Epoch: ", epoch+1)
    bulkmodel.train()
    bulkmodel.to(device)
    
    train_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        # adata_input = X_tensor.to(device)
        num_samples = mini_batch
        indices = torch.randperm(data.size(0))#[:num_samples]  # Generate random indices
        sampled_tensor = data[indices]

        adata_input = sampled_tensor.to(device)
        
        sc_mus, sc_logvars, sc_gmm_weights = scmodel.encode(adata_input)

        # sc_gmm_weights = sc_gmm_weights.sum(dim=0).reshape(1,-1)
        sc_mus_tensor = torch.stack(sc_mus)
        sc_logvars_tensor = torch.stack(sc_logvars)
        
        concatenated_sc_mus = sc_mus_tensor.sum(1, keepdim=True)
        concatenated_sc_logvars = sc_logvars_tensor.sum(1, keepdim=True)

        data = torch.sum(data, dim=0).reshape(1,-1)

        data = data.to(device)
        

        # pdb.set_trace()
        bulk_mus, bulk_logvars, bulk_gmm_weights = bulkmodel(data)
        bulk_mus = torch.stack(bulk_mus)
        if torch.isnan(bulk_mus).any():
            pdb.set_trace()
        bulk_logvars = torch.stack(bulk_logvars)
        # pdb.set_trace()


        mus_loss = nn.MSELoss()(sc_mus_tensor, bulk_mus.expand(paired_dataset.z_dim, mini_batch, paired_dataset.z_dim))
        logvars_loss = nn.MSELoss()(sc_logvars_tensor, bulk_logvars.expand(paired_dataset.z_dim, mini_batch, paired_dataset.z_dim))
        gmm_weights_loss = nn.MSELoss()(bulk_gmm_weights.expand(mini_batch, paired_dataset.z_dim), sc_gmm_weights)
        
        combined_loss = mus_loss + logvars_loss + gmm_weights_loss
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

    if (epoch+1)%100 == 0:
        bulkmodel = bulkmodel.eval()
        torch.save(bulkmodel.cpu().state_dict(), f"bulk_model_{epoch+1}.pt")
    

if __name__ == "__main__":
    train_loader = paired_dataset.dataloader
    bulk_model = models.bulkVAE(input_dim, hidden_dim, z_dim)
    optimizer = torch.optim.Adam(bulk_model.parameters(), lr=5e-3, weight_decay=10)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    scmodel = models.scVAE(input_dim, hidden_dim, z_dim)

    # Load the state dictionary and modify the keys
    state_dict = torch.load('sc_model_1200.pt')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    scmodel.load_state_dict(new_state_dict)
    scmodel = scmodel.to(device)

    for param in scmodel.parameters():
        param.requires_grad = False

    bulk_model = bulk_model.to(device)

    print("Loaded model.")

    for epoch in range(0, epochs + 1):

        
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