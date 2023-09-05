import torch
import torch.nn.functional as F
import pdb
import torch.nn as nn

def gmm_loss(gmm_weights_param, true_gmm_fractions=None, device='cuda'):
    # Default values for the true GMM fractions if not provided
    if true_gmm_fractions is None:
        true_gmm_fractions = torch.tensor([0.187, 0.129, 0.113, 0.113, 0.146, 0.193, 0.10, 0.013, 0.006])

    # Move true_gmm_fractions to the specified device
    true_gmm_fractions = true_gmm_fractions.to(device)
    
    # Apply softmax to gmm_weights_param
    # gmm_weights = torch.nn.Softmax(dim=0)(gmm_weights_param)
    
    # Calculate the MSE loss
    loss = nn.MSELoss()(gmm_weights_param, true_gmm_fractions)
    
    return loss


def KLDiv(mus, logvars, gmm_weights_param):
    KLD = 0
    for i in range(len(mus)):
        mu = mus[i]
        logvar = logvars[i]
        weight = gmm_weights_param[i]
        KLD += weight * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def bulk_loss(sc_mus, sc_logvars, bulk_mus, bulk_logvars):
    num_gmms = len(sc_mus)
    loss = 0
    for i in range(num_gmms):
        loss += nn.MSELoss()(sc_mus[i], bulk_mus[i]) + nn.MSELoss()(sc_logvars[i], bulk_logvars[i])
    return loss