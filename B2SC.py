import models
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
z_dim = data.z_dim


def sample_neuron(gmm_weights):
    num_neurons = len(gmm_weights)

    gmm_weights_np = gmm_weights.cpu().detach().numpy()

    # Generate a random neuron index based on the weights
    selected_neuron = np.random.choice(num_neurons, p=gmm_weights_np)

    return selected_neuron


def generate(b2sc_model, loader):
    b2sc_model.eval()
    recon_counts = []
    labels = []
    
    for batch_idx, (data_o,_) in enumerate(loader):
        data_o = data_o.to(device)
        
        # Shuffle the rows of the data tensor
        indices = torch.randperm(data_o.size(0))
        shuffled_data = data_o[indices]

        # Split the shuffled data into two halves
        mid_idx = shuffled_data.size(0) // 2
        first_half, second_half = shuffled_data[:mid_idx], shuffled_data[mid_idx:]

        # Sum the rows for each half
        data1 = torch.sum(first_half, dim=0).reshape(-1, input_dim)
        data2 = torch.sum(second_half, dim=0).reshape(-1, input_dim)
        # concatenate the two halves
        data = torch.cat((data1, data2), dim=0)

        data = data.to(device)

        b2sc_model = b2sc_model.to(device)
        # import pdb;pdb.set_trace()
        mus, logvars, gmm_weights = b2sc_model.encode(data)
        
        gmm_weights = gmm_weights.mean(0)
        gmm_weights = torch.nn.functional.softmax(gmm_weights, dim=0)
        # print(gmm_weights)
        
        selected_neuron = sample_neuron(gmm_weights)
        # print(selected_neuron)
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
    scmodel_state_dict = torch.load('model_1000.pt', map_location=device)
    bulk_model_state_dict = torch.load('bulk_model_1000.pt', map_location=device)

    # Modify the keys in the state dictionary to remove the "module." prefix
    scmodel_state_dict = {k.replace('module.', ''): v for k, v in scmodel_state_dict.items()}
    bulk_model_state_dict = {k.replace('module.', ''): v for k, v in bulk_model_state_dict.items()}

    def modify_keys(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            # Modify keys for fcs, bns, fc_means, and fc_logvars
            for idx in range(1, 10):  # Assuming 9 GMMs
                k = k.replace(f"fc{idx}.", f"fcs.{idx-1}.")
                k = k.replace(f"bn{idx}.", f"bns.{idx-1}.")
                k = k.replace(f"fc{idx}_mean.", f"fc_means.{idx-1}.")
                k = k.replace(f"fc{idx}_logvar.", f"fc_logvars.{idx-1}.")
            new_state_dict[k] = v
        return new_state_dict

    bulk_model_state_dict = modify_keys(bulk_model_state_dict)

    # Load the state dictionaries into the models
    bulk_model.load_state_dict(bulk_model_state_dict)
    scmodel.load_state_dict(scmodel_state_dict)

    # Transfer scmodel's decode to b2sc_model's decode
    b2sc_model.fc_d1.weight.data = scmodel.fc_d1.weight.data.clone()
    b2sc_model.fc_d1.bias.data = scmodel.fc_d1.bias.data.clone()

    b2sc_model.bn_d1.weight.data = scmodel.bn_d1.weight.data.clone()
    b2sc_model.bn_d1.bias.data = scmodel.bn_d1.bias.data.clone()

    b2sc_model.fc_d2.weight.data = scmodel.fc_d2.weight.data.clone()
    b2sc_model.fc_d2.bias.data = scmodel.fc_d2.bias.data.clone()

    b2sc_model.bn_d2.weight.data = scmodel.bn_d2.weight.data.clone()
    b2sc_model.bn_d2.bias.data = scmodel.bn_d2.bias.data.clone()

    b2sc_model.fc_d3.weight.data = scmodel.fc_d3.weight.data.clone()
    b2sc_model.fc_d3.bias.data = scmodel.fc_d3.bias.data.clone()

    b2sc_model.bn_d3.weight.data = scmodel.bn_d3.weight.data.clone()
    b2sc_model.bn_d3.bias.data = scmodel.bn_d3.bias.data.clone()

    b2sc_model.fc_count.weight.data = scmodel.fc_count.weight.data.clone()
    b2sc_model.fc_count.bias.data = scmodel.fc_count.bias.data.clone()

    # Transfer fc_gmm_weights from bulk_model to b2sc_model
    b2sc_model.fc_gmm_weights.weight.data = bulk_model.fc_gmm_weights.weight.data.clone()
    b2sc_model.fc_gmm_weights.bias.data = bulk_model.fc_gmm_weights.bias.data.clone()

    # Transfer bulk_model's encode to b2sc_model's encode
    for i in range(b2sc_model.num_gmms):
        b2sc_model.fcs[i].weight.data = bulk_model.fcs[i].weight.data.clone()
        b2sc_model.fcs[i].bias.data = bulk_model.fcs[i].bias.data.clone()

        b2sc_model.bns[i].weight.data = bulk_model.bns[i].weight.data.clone()
        b2sc_model.bns[i].bias.data = bulk_model.bns[i].bias.data.clone()

        b2sc_model.fc_means[i].weight.data = bulk_model.fc_means[i].weight.data.clone()
        b2sc_model.fc_means[i].bias.data = bulk_model.fc_means[i].bias.data.clone()

        b2sc_model.fc_logvars[i].weight.data = bulk_model.fc_logvars[i].weight.data.clone()
        b2sc_model.fc_logvars[i].bias.data = bulk_model.fc_logvars[i].bias.data.clone()

    b2sc_model.fc_gmm_weights.weight.data = bulk_model.fc_gmm_weights.weight.data.clone()
    b2sc_model.fc_gmm_weights.bias.data = bulk_model.fc_gmm_weights.bias.data.clone()

    # # Transfer encoder weights from bulkVAE to B2SC
    # encoder_layers = ['fcs', 'bns', 'fc_means', 'fc_logvars']
    # for idx in range(bulk_model.num_gmms):
    #     for layer in encoder_layers:
    #         getattr(b2sc_model, layer)[idx].load_state_dict(
    #             getattr(bulk_model, layer)[idx].state_dict())

    # # Transfer decoder weights from scVAE to B2SC
    # decoder_layers = ['fc_d1', 'bn_d1', 'dropout_d1', 'fc_d2', 'bn_d2', 'dropout_d2', 'fc_d3', 'bn_d3', 'dropout_d3', 'fc_count']
    # for layer_name in decoder_layers:
    #     getattr(b2sc_model, layer_name).load_state_dict(
    #         getattr(scmodel, layer_name).state_dict())

    # Take first batch from train_loader.
    data1, _ = next(iter(data.loader))
    data1 = data1.to(device)
    gmm_weights = b2sc_model.encode(data1)[-1]
    print(gmm_weights)


    print("Loaded models")
    aggregate_recon_counts = []
    aggregate_labels = []

    for i in range(data.mini_batch):
        if (i+1)%100 == 0:
            print(f"Generating batch {i+1} / {data.mini_batch}")
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

    # Save to file
    np.save('recon_counts.npy', recon_counts_np)
    # Save labels list as numpy array
    np.save('labels.npy', np.array(aggregate_labels))

