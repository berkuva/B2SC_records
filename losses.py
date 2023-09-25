import torch
import torch.nn.functional as F
import pdb
import torch.nn as nn
from torch.distributions import Distribution, Poisson, NegativeBinomial
import numpy as np

class PoissonDist(Distribution):
    def __init__(self, rate, validate_args=None):
        self._poisson = Poisson(rate, validate_args)
        self._rate = rate
    
    def log_prob(self, value):
        return self._poisson.log_prob(value)


class ZIP(Distribution):
    def __init__(self, rate, logits, validate_args=None):
        self._poisson = Poisson(rate, validate_args)
        self._rate = rate
        self._logits = logits
        self._gate = torch.nn.Sigmoid()
        self._epsilon = 1e-6  # Define a small epsilon
    
    def log_prob(self, value):
        gate = self._gate(self._logits)
        
        # Add epsilon inside the log to prevent log(0)
        return torch.where(value == 0,
                           torch.log(gate + (1-gate) * torch.exp(self._poisson.log_prob(value)) + self._epsilon),
                           self._poisson.log_prob(value))


class NBDist(Distribution):
    def __init__(self, total_count, probs, validate_args=None):
        self._nb = NegativeBinomial(total_count, probs, validate_args)
        self._total_count = total_count
        self._probs = probs
    
    def log_prob(self, value):
        return self._nb.log_prob(value)


class ZINB(Distribution):
    def __init__(self, total_count, probs, logits, validate_args=None):
        self._nb = torch.distributions.NegativeBinomial(total_count, probs, validate_args)
        self._total_count = total_count
        self._probs = probs
        self._logits = logits
        self._gate = torch.nn.Sigmoid()
    
    def log_prob(self, value):
        gate = self._gate(self._logits)
        return torch.where(value == 0,
                           torch.log(gate + (1-gate) * torch.exp(self._nb.log_prob(value))),
                           self._nb.log_prob(value))
    


def zinb_loss(preds, targets, zero_inflation_prob, theta):
    """
    Computes the ZINB loss.

    Args:
    - preds: Predicted means of the NB distribution.
    - targets: True counts.
    - zero_inflation_prob: Predicted probability of zero inflation.
    - theta: Dispersion parameter for the NB distribution.

    Returns:
    - ZINB loss
    """
    # Ensure values are in safe ranges
    preds = torch.clamp(preds, min=1e-8)
    theta = torch.clamp(theta, min=1e-8)
    
    # Zero-Inflated part
    zero_loss = - (targets * torch.log(zero_inflation_prob + 1e-8) + (1. - targets) * torch.log(1. - zero_inflation_prob + 1e-8))
    
    # NB likelihood
    nb_loss = (torch.lgamma(targets + 1/theta) - torch.lgamma(targets + 1) - torch.lgamma(1/theta)
            + 1/theta * torch.log(1/theta + 1e-8) + 1/theta * torch.log(1 + theta * preds + 1e-8)
            - targets * torch.log(theta * preds + 1e-8) - targets * torch.log(1 + theta * preds + 1e-8))
    
    nb_loss = -nb_loss

    return zero_loss.mean() + nb_loss.mean()


def gmm_loss(gmm_weights, true_gmm_fractions=None, device='cuda'):
    # Default values for the true GMM fractions if not provided
    if true_gmm_fractions is None:
        true_gmm_fractions = torch.tensor([0.187, 0.129, 0.113, 0.113, 0.146, 0.193, 0.10, 0.013, 0.006])

    # Move true_gmm_fractions to the specified device
    true_gmm_fractions = true_gmm_fractions.to(device)
    
    batch_size = gmm_weights.size(0)
    true_gmm_fractions_expanded = true_gmm_fractions.unsqueeze(0).repeat(batch_size, 1)

    # Calculate the MSE loss
    loss = nn.MSELoss()(gmm_weights, true_gmm_fractions_expanded)
    
    return loss


def KLDiv(mus, logvars, gmm_weights):
    KLD = 0
    for i in range(len(mus)):
        mu = mus[i]
        logvar = logvars[i]
        weight = gmm_weights[i]
        KLD += weight * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD.sum()


def bulk_loss(sc_mus, sc_logvars, bulk_mus, bulk_logvars):
    num_gmms = len(sc_mus)
    mse_loss = nn.MSELoss()
    loss = 0
    for i in range(num_gmms):
        loss += mse_loss(sc_mus[i], bulk_mus[i]) + mse_loss(sc_logvars[i], bulk_logvars[i])
    return loss
