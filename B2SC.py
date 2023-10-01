import models
import paired_dataset
from paired_dataset import *
import numpy as np
import torch
# import scanpy as sc
import time
import pdb

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"

# Define hyperparameters and other settings
input_dim = paired_dataset.input_dim
hidden_dim = paired_dataset.hidden_dim
z_dim = paired_dataset.z_dim


# def sample_neuron(gmm_weights):
#     num_neurons = len(gmm_weights)
#     gmm_weights_np = gmm_weights.cpu().detach().numpy()
#     # Generate a random neuron index based on the weights
#     selected_neuron = np.random.choice(num_neurons, p=gmm_weights_np)
#     return selected_neuron

# def sample_neuron(gmm_weights):
#     probabilities = torch.nn.functional.softmax(gmm_weights, dim=0)
#     probabilities = probabilities / (probabilities.sum() + 1e-10)
#     return torch.multinomial(probabilities, 1).item()


def generate(b2sc_model, loader):
    b2sc_model.eval()
    recon_counts = []
    labels = []
    
    for batch_idx, (data_o,_) in enumerate(loader):
        data_o = data_o.to(device)

        data = torch.sum(data_o, dim=0).reshape(1,-1)

        data = data.to(device)
        
        b2sc_model = b2sc_model.to(device)
        # import pdb;pdb.set_trace()
        recon_count, selected_neuron = b2sc_model(data, variability=1)

        
        labels.append(selected_neuron)
        recon_counts.append(recon_count)

    return recon_counts, labels
    

if __name__ == "__main__":
    # Instantiate models
    scmodel = models.scVAE(input_dim, hidden_dim, z_dim).to(device)
    bulk_model = models.bulkVAE(input_dim, hidden_dim, z_dim).to(device)
    b2sc_model = models.B2SC(input_dim, hidden_dim, z_dim).to(device)
    
    # Load state dictionaries
    scmodel_state_dict = torch.load('/u/hc2kc/scVAE/sc_model_700.pt', map_location=device)
    bulk_model_state_dict = torch.load('/u/hc2kc/scVAE/bulk_model_2500.pt', map_location=device)

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

    # b2sc_model.bn_d1.weight.data = scmodel.bn_d1.weight.data.clone()
    # b2sc_model.bn_d1.bias.data = scmodel.bn_d1.bias.data.clone()

    b2sc_model.fc_d2.weight.data = scmodel.fc_d2.weight.data.clone()
    b2sc_model.fc_d2.bias.data = scmodel.fc_d2.bias.data.clone()

    # b2sc_model.bn_d2.weight.data = scmodel.bn_d2.weight.data.clone()
    # b2sc_model.bn_d2.bias.data = scmodel.bn_d2.bias.data.clone()

    b2sc_model.fc_d3.weight.data = scmodel.fc_d3.weight.data.clone()
    b2sc_model.fc_d3.bias.data = scmodel.fc_d3.bias.data.clone()

    # b2sc_model.bn_d3.weight.data = scmodel.bn_d3.weight.data.clone()
    # b2sc_model.bn_d3.bias.data = scmodel.bn_d3.bias.data.clone()

    # b2sc_model.bns.weight.data = scmodel.bns.weight.data.clone()
    # b2sc_model.bns.bias.data = scmodel.bns.bias.data.clone()

    b2sc_model.fc_count.weight.data = scmodel.fc_mean.weight.data.clone()
    b2sc_model.fc_count.bias.data = scmodel.fc_mean.bias.data.clone()

    # Transfer fc_gmm_weights from bulk_model to b2sc_model
    b2sc_model.fc_gmm_weights.weight.data = bulk_model.fc_gmm_weights.weight.data.clone()
    b2sc_model.fc_gmm_weights.bias.data = bulk_model.fc_gmm_weights.bias.data.clone()

    # Transfer bulk_model's encode to b2sc_model's encode
    for i in range(b2sc_model.num_gmms):
        b2sc_model.fcs[i].weight.data = bulk_model.fcs[i].weight.data.clone()
        b2sc_model.fcs[i].bias.data = bulk_model.fcs[i].bias.data.clone()

        b2sc_model.fc_means[i].weight.data = bulk_model.fc_means[i].weight.data.clone()
        b2sc_model.fc_means[i].bias.data = bulk_model.fc_means[i].bias.data.clone()

        b2sc_model.fc_logvars[i].weight.data = bulk_model.fc_logvars[i].weight.data.clone()
        b2sc_model.fc_logvars[i].bias.data = bulk_model.fc_logvars[i].bias.data.clone()

        # b2sc_model.bns[i].weight.data = scmodel.bns[i].weight.data.clone()
        # b2sc_model.bns[i].bias.data = scmodel.bns[i].bias.data.clone()


    print("Loaded models")

    # pdb.set_trace()
    # b2sc_model.encode(X_tensor.to(device).sum(0))[-1]
    # tensor([0.1868, 0.1287, 0.1126, 0.1125, 0.1461, 0.1928, 0.1002, 0.0130, 0.0059],
    #     device='cuda:0', grad_fn=<AddBackward0>)

    all_recon_counts = []
    all_labels = []
    

    num_runs =2700  # or however many times to run the generation

    for i in range(num_runs):
        if (i+1)%100==0:
            print(f"Run: {i+1}")
            # Save to file
            # recon_count_tensor = np.array(all_recon_counts)
            # labels_tensor = np.array(all_labels)
            # np.save('recon_counts.npy', np.array(recon_count_tensor))
            # np.save('labels.npy', np.array(labels_tensor))

        recon_counts, labels = generate(b2sc_model, paired_dataset.dataloader)
        for k in range(len(recon_counts)):
            all_recon_counts.append(recon_counts[k].cpu().detach().numpy().tolist())
            all_labels.append(labels[k])
    
    # all_recon_counts = np.load('recon_counts.npy')
    # all_labels = np.load('labels.npy')
    all_recon_counts = np.array(all_recon_counts)
    all_labels = np.array(all_labels)


    mean_variance_dict = {label.item(): (paired_dataset.X_tensor[paired_dataset.cell_types_tensor == label].mean(dim=0),\
                                        paired_dataset.X_tensor[paired_dataset.cell_types_tensor == label].var(dim=0, unbiased=True))\
                        for label in torch.unique(paired_dataset.cell_types_tensor)}

    # 
    # Convert labels to a tensor if it's a numpy array
    labels_tensor = torch.tensor(all_labels)

    # Unique labels
    unique_labels = torch.unique(labels_tensor)

    for label in unique_labels:
        label = label.item()
        label_indices = (all_labels == label)
        label_data = all_recon_counts[label_indices]
        
        desired_mean, desired_variance = mean_variance_dict[label]
        
        # If your desired_mean and desired_variance are torch tensors, convert them to numpy arrays
        desired_mean = desired_mean.numpy() if isinstance(desired_mean, torch.Tensor) else desired_mean
        desired_variance = desired_variance.numpy() if isinstance(desired_variance, torch.Tensor) else desired_variance
        
        current_mean = np.mean(label_data, axis=0)
        # pdb.set_trace()
        current_variance = np.var(label_data, axis=0, ddof=1)  # ddof=1 for unbiased variance
        
        # Avoid division by zero by replacing zero variance with a small value
        current_variance = np.where(current_variance == 0, np.finfo(float).eps, current_variance)
        
        scale_factor = np.sqrt(desired_variance / current_variance)
        shift_value = desired_mean - current_mean * scale_factor
        
        # Print statements for verification
        print(f'Label: {label}')
        print(f'Current Mean: {current_mean}, Desired Mean: {desired_mean}')
        print(f'Current Variance: {current_variance}, Desired Variance: {desired_variance}')
        print(f'Scale Factor: {scale_factor}')
        print(f'Shift Value: {shift_value}')
        
        # Update data for the current label
        all_recon_counts[label_indices] = label_data * scale_factor + shift_value

        # Verification after adjustment
        adjusted_mean = np.mean(all_recon_counts[label_indices], axis=0)
        adjusted_variance = np.var(all_recon_counts[label_indices], axis=0, ddof=1)
        print(f'Adjusted Mean: {adjusted_mean}')
        print(f'Adjusted Variance: {adjusted_variance}')
        print('-' * 50)  # separator line


    # Clipping negative values to 0
    all_recon_counts = np.clip(all_recon_counts, 0, None)

    # Converting values to integers
    all_recon_counts = all_recon_counts.astype(np.int)

    # Save updated all_recon_counts and labels to disk
    np.save('recon_counts.npy', all_recon_counts)
    np.save('labels.npy', all_labels)


    # import pdb;pdb.set_trace()

    # # set all_recon_counts equal to saved recon_counts.npy.
    # try:
    #     all_recon_counts = np.load('recon_counts.npy').tolist()
    #     all_labels = np.load('labels.npy').tolist()
    # except:
    #     all_recon_counts = []
    #     all_labels = []

    # howmany = 500
    # for l in range(howmany):
    #     print(f"Run: {l+1}")
    #     recon_counts, labels = generate_desired_celltypes(6, b2sc_model, 1)
    #     for k in range(len(recon_counts)):
    #         all_recon_counts.append(recon_counts[k].cpu().detach().numpy().tolist())
    #         all_labels.append(labels[k])
    #     if (l+1)%10==0:
    #         recon_count_tensor = np.array(all_recon_counts)
    #         labels_tensor = np.array(all_labels)

    #         # Save to file
    #         np.save('recon_counts.npy', recon_count_tensor)
    #         np.save('labels.npy', labels_tensor)
            # all_recon_counts = []
            # all_labels = []