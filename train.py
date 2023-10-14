import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def compute_gmm_loss(gmm_weights, args):
    # Default values for the true GMM fractions if not provided
    true_gmm_fractions = torch.FloatTensor(args.cell_type_fractions)

    # Move true_gmm_fractions to the specified device
    true_gmm_fractions = true_gmm_fractions.to(args.device)
    
    batch_size = gmm_weights.size(0)
    true_gmm_fractions_expanded = true_gmm_fractions.unsqueeze(0).repeat(batch_size, 1)

    # Calculate the MSE loss
    loss = nn.MSELoss()(gmm_weights, true_gmm_fractions_expanded)
    
    return loss

# Training Loop with warm-up
def train(epoch,
          model,
          optimizer,
          gmm_weights_backup,
          args,
          train_gmm_till=500,
          print_every=100,
          save_every=500):
    print(f'Epoch: {epoch+1}')
    model.train()

    train_loss = 0
    
    for batch_idx, (data,labels) in enumerate(args.dataloader):
        data = data.to(args.device)
        # model = nn.DataParallel(model).to(device)
        labels = labels.to(args.device)
        optimizer.zero_grad()

        preds, zero_inflation_prob, theta, mus, logvars, gmm_weights = model(data)
        
        if train_gmm_till > epoch+1:
            loss = compute_gmm_loss(gmm_weights, args)
        else:
            ceL = nn.CrossEntropyLoss()(preds, data)
            loss =  ceL
        
        if train_gmm_till == epoch+1:
            gmm_weights_backup = {name: param.clone() for name, param in model.fc_gmm_weights.named_parameters()}
            for param in model.fc_gmm_weights.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)

        loss.backward()
        optimizer.step()

        if epoch+1 > train_gmm_till:
            for name, param in model.fc_gmm_weights.named_parameters():
                param.data = gmm_weights_backup[name]


    if (epoch+1)%print_every == 0:
        print(f'====> Epoch: {epoch+1} Average loss: {train_loss / len(args.dataloader.dataset):.4f}')

    if (epoch+1)%save_every == 0:
        torch.save(model.cpu().state_dict(), f"sc_model_{epoch+1}.pt")
        print("---")
        print(preds[0])
        model.to(args.device)
        model.eval()
        
        with torch.no_grad():
            # recon_batch, mus, logvars, gmm_weights = model(X_tensor)
            mus, logvars, gmm_weights = model.encode(args.X_tensor.to(args.device))
            z = model.reparameterize(mus, logvars, gmm_weights).cpu().numpy()

            # Convert cell_types_tensor to numpy for plotting
            raw_labels = args.cell_types_tensor.numpy()

            # Convert raw_labels to their names using the label_map
            label_names = np.array([args.label_map[str(label)] for label in raw_labels])
            unique_labels = np.unique(label_names)

            # Set up t-SNE
            tsne = TSNE()
            embedding = tsne.fit_transform(z)
            

            # Plot the t-SNE representation
            plt.figure(figsize=(10, 10))
            for label in unique_labels:
                indices = np.where(label_names == label)
                plt.scatter(embedding[indices, 0], embedding[indices, 1], color=args.color_map[label], label=label)
            # move legend to right side outside of the figure.
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.savefig(f"tsne_{epoch+1}.png")
            

    return gmm_weights_backup


if __name__ == "__main__":
    from configure import configure
    from models import scVAE
    

    data_dir = "/u/hc2kc/scVAE/pbmc1k/data/" # "path_to_data_directory"  # Please provide the correct path here
    barcode_path = data_dir+'barcode_to_celltype.csv' #"path_to_barcode_path"  # And also the correct path here

    args = configure(data_dir, barcode_path)

    # Create VAE and optimizer
    model = scVAE(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Weight initialization
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight) 
            m.bias.data.fill_(0.01)

    model.apply(init_weights) 
    model = model.to(args.device)

    # Clamp weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.clamp_(-5, 5)

    gmm_weights_backup = None
    for epoch in range(0, args.train_epochs + 1):
        gmm_weights_backup = train(epoch,
                                   model,
                                   optimizer,
                                   gmm_weights_backup,
                                   args,
                                   train_gmm_till=500)
