import numpy as np
import torch


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"


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
    from configure import configure
    from models import scVAE, bulkVAE, B2SC
    

    data_dir = "/u/hc2kc/scVAE/pbmc1k/data/" # "path_to_data_directory"  # Please provide the correct path here
    barcode_path = data_dir+'barcode_to_celltype.csv' #"path_to_barcode_path"  # And also the correct path here

    args = configure(data_dir, barcode_path)


    bulk_model = bulkVAE(args)
    optimizer = torch.optim.Adam(bulk_model.parameters(), lr=args.learning_rate)

    scmodel = scVAE(args)

    # Instantiate models
    scmodel = scVAE(args).to(device)
    bulk_model = bulkVAE(args).to(device)
    b2sc_model = B2SC(args).to(device)
    
    # Load state dictionaries
    scmodel_state_dict = torch.load('/u/hc2kc/scVAE/general/sc_model_50.pt', map_location=device) # Please provide the correct path here
    bulk_model_state_dict = torch.load('/u/hc2kc/scVAE/general/bulk_model_30.pt', map_location=device) # Please provide the correct path here

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

    b2sc_model.fc_d2.weight.data = scmodel.fc_d2.weight.data.clone()
    b2sc_model.fc_d2.bias.data = scmodel.fc_d2.bias.data.clone()

    b2sc_model.fc_d3.weight.data = scmodel.fc_d3.weight.data.clone()
    b2sc_model.fc_d3.bias.data = scmodel.fc_d3.bias.data.clone()

    b2sc_model.fc_count.weight.data = scmodel.fc_mean.weight.data.clone()
    b2sc_model.fc_count.bias.data = scmodel.fc_mean.bias.data.clone()

    # Transfer fc_gmm_weights from bulk_model to b2sc_model
    b2sc_model.fc_gmm_weights.weight.data = bulk_model.fc_gmm_weights.weight.data.clone()
    b2sc_model.fc_gmm_weights.bias.data = bulk_model.fc_gmm_weights.bias.data.clone()

    # Transfer bulk_model's encode to b2sc_model's encode
    for i in range(args.z_dim):
        b2sc_model.fcs[i].weight.data = bulk_model.fcs[i].weight.data.clone()
        b2sc_model.fcs[i].bias.data = bulk_model.fcs[i].bias.data.clone()

        b2sc_model.fc_means[i].weight.data = bulk_model.fc_means[i].weight.data.clone()
        b2sc_model.fc_means[i].bias.data = bulk_model.fc_means[i].bias.data.clone()

        b2sc_model.fc_logvars[i].weight.data = bulk_model.fc_logvars[i].weight.data.clone()
        b2sc_model.fc_logvars[i].bias.data = bulk_model.fc_logvars[i].bias.data.clone()

    print("Loaded models")

    all_recon_counts = []
    all_labels = []
    

    num_runs = args.batch_size # or however many times to run the generation

    for i in range(num_runs):
        if (i+1)%100==0:
            print(f"Run: {i+1}")
            # Save to file
            recon_count_tensor = np.array(all_recon_counts)
            labels_tensor = np.array(all_labels)
            np.save('recon_counts.npy', np.array(recon_count_tensor))
            np.save('labels.npy', np.array(labels_tensor))

        recon_counts, labels = generate(b2sc_model, args.dataloader)
        for k in range(len(recon_counts)):
            all_recon_counts.append(recon_counts[k].cpu().detach().numpy().tolist())
            all_labels.append(labels[k])
    

    all_recon_counts = all_recon_counts.astype(np.int)

    np.save('recon_counts_l1_sc.npy', all_recon_counts)
    np.save('labels_l1_sc.npy', all_labels)



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