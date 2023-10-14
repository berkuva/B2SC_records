import torch
import torch.nn as nn


def train(epoch,
          bulkmodel,
          optimizer,
          sc_mus,
          sc_logvars,
          sc_gmm_weights,
          args,
          gmm_weights_backup,
          train_gmm_till=1000,
          train_mu_till=2000,
          print_every=200,
          save_every=500):
    
    # print("Epoch: ", epoch+1)
    bulkmodel.train()
    bulkmodel.to(args.device)
    
    train_loss = 0
    for batch_idx, (data,_) in enumerate(args.dataloader):

        data = torch.sum(data, dim=0).reshape(1,-1)
        data = data.to(args.device)

        sc_mus_tensor = torch.stack(sc_mus)
        sc_logvars_tensor = torch.stack(sc_logvars)
        
        concatenated_sc_mus = sc_mus_tensor.sum(1, keepdim=True)
        concatenated_sc_logvars = sc_logvars_tensor.sum(1, keepdim=True)

        
        bulk_mus, bulk_logvars, bulk_gmm_weights = bulkmodel(data)
        bulk_mus = torch.stack(bulk_mus)
        bulk_logvars = torch.stack(bulk_logvars)


        if epoch+1 <= train_gmm_till:
            gmm_weights_loss = nn.MSELoss()(bulk_gmm_weights.expand(len(sc_gmm_weights), args.z_dim), sc_gmm_weights)
            combined_loss = gmm_weights_loss
        else:
            gmm_weights_backup = {name: param.clone() for name, param in bulkmodel.fc_gmm_weights.named_parameters()}
            for param in bulkmodel.fc_gmm_weights.parameters():
                param.requires_grad = False
            
            if epoch+1 < train_mu_till:
                mus_loss = nn.MSELoss()(concatenated_sc_mus, bulk_mus.expand(args.z_dim, concatenated_sc_mus.shape[1], args.z_dim))
                combined_loss = mus_loss
            else:
                mu_weights_backup = {name: param.clone() for name, param in bulkmodel.fc_means.named_parameters()}
                for param in bulkmodel.fc_means.parameters():
                    param.requires_grad = False
                logvars_loss = nn.MSELoss()(concatenated_sc_logvars, bulk_logvars.expand(args.z_dim, concatenated_sc_logvars.shape[1], args.z_dim))
                combined_loss = logvars_loss
            
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, bulkmodel.parameters()), lr=1e-4)

        combined_loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        if epoch+1 > train_gmm_till:
            for name, param in bulkmodel.fc_gmm_weights.named_parameters():
                param.data = gmm_weights_backup[name]
        if epoch+1 > train_mu_till:
            for name, param in bulkmodel.fc_means.named_parameters():
                param.data = mu_weights_backup[name]

        train_loss += combined_loss.item()

    if (epoch+1)%print_every==0:
        print(f'====> Epoch: {epoch+1} Average loss: {train_loss:.4f}')
        print(bulk_gmm_weights.mean(0))
        print(sc_gmm_weights.mean(0))
        
      
    if (epoch+1)%save_every == 0:
        bulkmodel = bulkmodel.eval()
        torch.save(bulkmodel.cpu().state_dict(), f"bulk_model_{epoch+1}.pt")
    
    return gmm_weights_backup
    
    
if __name__ == "__main__":
    from configure import configure
    from models import scVAE, bulkVAE
    

    data_dir = "/u/hc2kc/scVAE/pbmc1k/data/" # "path_to_data_directory"  # Please provide the correct path here
    barcode_path = data_dir+'barcode_to_celltype.csv' #"path_to_barcode_path"  # And also the correct path here

    args = configure(data_dir, barcode_path)


    bulk_model = bulkVAE(args)
    scmodel = scVAE(args)

    optimizer = torch.optim.Adam(bulk_model.parameters(), lr=args.learning_rate)

    # Load the state dictionary and modify the keys
    state_dict = torch.load('sc_model_50.pt') # Please provide the correct path here
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    scmodel.load_state_dict(new_state_dict)
    scmodel = scmodel.to(args.device)

    for param in scmodel.parameters():
        param.requires_grad = False
    

    # Transfer weights for the GMM weights Linear layer
    bulk_model.fc_gmm_weights.weight.data.copy_(scmodel.fc_gmm_weights.weight.data)
    bulk_model.fc_gmm_weights.bias.data.copy_(scmodel.fc_gmm_weights.bias.data)

    sc_mus, sc_logvars, sc_gmm_weights = scmodel.encode(args.X_tensor.to(args.device))

    bulk_model = bulk_model.to(args.device)
    print("Loaded model.")
    

    for sc_fc, bulk_fc in zip(scmodel.fcs, bulk_model.fcs):
        bulk_fc.weight.data.copy_(sc_fc.weight.data)
        bulk_fc.bias.data.copy_(sc_fc.bias.data)

    # Transfer weights for the mean Linear layers
    for sc_fc_mean, bulk_fc_mean in zip(scmodel.fcs_mean, bulk_model.fc_means):
        bulk_fc_mean.weight.data.copy_(sc_fc_mean.weight.data)
        bulk_fc_mean.bias.data.copy_(sc_fc_mean.bias.data)

    # Transfer weights for the log variance Linear layers
    for sc_fc_logvar, bulk_fc_logvar in zip(scmodel.fcs_logvar, bulk_model.fc_logvars):
        bulk_fc_logvar.weight.data.copy_(sc_fc_logvar.weight.data)
        bulk_fc_logvar.bias.data.copy_(sc_fc_logvar.bias.data)


    gmm_weights_backup = None
    for epoch in range(0, args.bulk_epochs+1):
                
        gmm_weights_backup = train(epoch,
                                    bulk_model,
                                    optimizer,
                                    sc_mus,
                                    sc_logvars,
                                    sc_gmm_weights,
                                    args,
                                    gmm_weights_backup)
