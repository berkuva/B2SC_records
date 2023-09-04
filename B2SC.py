import models
import losses
import data
from data import *
import numpy as np
import torch
import scanpy as sc


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"

# Define hyperparameters and other settings
input_dim = 32738
hidden_dim = 700
z_dim = 9
epochs = 2000


def leiden_clustering(adata):
    sc.pp.pca(adata, n_comps=50)  # Compute the top 50 principal components
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=10)  # Use PCA representation to compute neighbors with 10 neighbors
    sc.tl.leiden(adata)
    return adata.obs['leiden']



def convert_to_adata(recon_counts, labels):
    # Convert recon_counts to numpy array
    recon_counts_np = recon_counts.cpu().detach().numpy()

    # Create AnnData object
    adata = sc.AnnData(X=recon_counts_np)

    # Add labels to adata object
    adata.obs['labels'] = np.array(labels).astype(str)  # Convert labels to string for proper mapping
    return adata


# def sample_neuron(gmm_weights):
#     gmm_weights_np = np.array(gmm_weights)
#     num_neurons = len(gmm_weights_np)
    
#     # Normalize the probabilities
#     gmm_weights_np = gmm_weights_np / gmm_weights_np.sum()
    
#     selected_neuron = np.random.choice(num_neurons, p=gmm_weights_np)
#     return selected_neuron


def sample_neuron(gmm_weights):
    num_neurons = len(gmm_weights)

    # Convert gmm_weights to numpy array
    # gmm_weights_np = gmm_weights.detach().numpy()

    gmm_weights_np = gmm_weights.cpu().detach().numpy()

    # Generate a random neuron index based on the weights
    selected_neuron = np.random.choice(num_neurons, p=gmm_weights_np)

    return selected_neuron


def generate(b2sc_model, loader):
    b2sc_model.eval()
    recon_counts = []
    labels = []
    
    for batch_idx, (data,_) in enumerate(loader):
        data = data.to(device)

        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]

        # Split the shuffled data into two halves
        mid_idx = shuffled_data.size(0) // 2
        first_half, second_half = shuffled_data[:mid_idx], shuffled_data[mid_idx:]

        # Sum the rows for each half
        data1 = torch.sum(first_half, dim=0).reshape(-1, input_dim)
        data2 = torch.sum(second_half, dim=0).reshape(-1, input_dim)
        data = torch.cat((data1, data2), dim=0)

        b2sc_model = b2sc_model.to(device)

        selected_neuron = sample_neuron(b2sc_model.gmm_weights)
        labels.append(selected_neuron)

        recon_count = b2sc_model(data, selected_neuron)

        # sys.exit if nan in recon_count.
        if torch.isnan(recon_count).any():
            print("Nan in recon_count")
            import sys; sys.exit()
        recon_counts.append(recon_count)

    return recon_counts, labels
    

if __name__ == "__main__":
    # Instantiate models
    scmodel = models.scVAE(input_dim, hidden_dim, z_dim).to(device)
    bulk_model = models.bulkVAE(input_dim, hidden_dim, z_dim).to(device)
    b2sc_model = models.B2SC(input_dim, hidden_dim, z_dim).to(device)

    # Load state dictionaries
    scmodel_state_dict = torch.load('Pmodel_1000.pt', map_location=device)
    bulk_model_state_dict = torch.load('bulk_model_400.pt', map_location=device)

    # Apply state dictionaries to the models
    scmodel.load_state_dict(scmodel_state_dict)
    bulk_model.load_state_dict(bulk_model_state_dict)

    # Transfer encoder weights from layers 1 to 9.
    for layer_name in ['fc1', 'bn1', 'fc1_mean', 'fc1_logvar',\
                       'fc2', 'bn2', 'fc2_mean', 'fc2_logvar', \
                        'fc3', 'bn3', 'fc3_mean', 'fc3_logvar', \
                        'fc4', 'bn4', 'fc4_mean', 'fc4_logvar', \
                        'fc5', 'bn5', 'fc5_mean', 'fc5_logvar', \
                        'fc6', 'bn6', 'fc6_mean', 'fc6_logvar', \
                        'fc7', 'bn7', 'fc7_mean', 'fc7_logvar', \
                        'fc8', 'bn8', 'fc8_mean', 'fc8_logvar', \
                        'fc9', 'bn9', 'fc9_mean', 'fc9_logvar']:
        getattr(b2sc_model, layer_name).load_state_dict(
            getattr(bulk_model, layer_name).state_dict())

    # Transfer decoder weights
    for layer_name in ['fc_d1', 'bn_d1', 'dropout_d1', 'fc_d2', 'bn_d2', 'dropout_d2', 'fc_d3', 'bn_d3', 'dropout_d3', 'fc_count']:
        getattr(b2sc_model, layer_name).load_state_dict(
            getattr(scmodel, layer_name).state_dict())
    # pdb.set_trace()
    # scmodel = scmodel.to(device)
    bulk_model = bulk_model.to(device)
    
    b2sc_model = b2sc_model.to(device)
    b2sc_model.gmm_weights = bulk_model.gmm_weights
    print("Loaded models")
    aggregate_recon_counts = []
    aggregate_labels = []


    for i in range(1000):
        if (i+1)%100 == 0:
            print(f"Generating batch {i+1} / 1000")
        recon_counts, labels = generate(b2sc_model, data.loader)
        recon_count = recon_counts[0]
        label = labels[0]
        
        
        # for recon_count in recon_counts:
        aggregate_recon_counts.append(recon_count)
        # for label in labels:
        aggregate_labels.append(label)
    print("Generated counts")
    
    recon_counts_tensor = torch.stack(aggregate_recon_counts).squeeze()

    # Move to CPU
    recon_counts_tensor = recon_counts_tensor.cpu()

    # Convert to numpy array
    recon_counts_np = recon_counts_tensor.detach().numpy()

    # leiden_clusters = leiden_clustering(convert_to_adata(recon_counts_tensor, aggregate_labels))
    # # Save leiden clusters
    # np.save('leiden_clusters2.npy', leiden_clusters)

    # ars = adjusted_rand_score(aggregate_labels, leiden_clusters)
    # print(f"Adjusted Rand Score:", {ars})

    # Save to file
    np.save('recon_counts.npy', recon_counts_np)
    # Save labels list as numpy array
    np.save('labels.npy', np.array(aggregate_labels))

