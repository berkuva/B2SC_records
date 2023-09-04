import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Distribution, Poisson, NegativeBinomial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu" 
print(device)
import pdb


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


class scVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout_rate=0.1):
        super(scVAE, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # GMM1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1_mean = nn.Linear(hidden_dim, z_dim)
        self.fc1_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM2
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, z_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM3
        self.fc3 = nn.Linear(input_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, z_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM4
        self.fc4 = nn.Linear(input_dim, hidden_dim)        
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc4_mean = nn.Linear(hidden_dim, z_dim)
        self.fc4_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM5
        self.fc5 = nn.Linear(input_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.fc5_mean = nn.Linear(hidden_dim, z_dim)
        self.fc5_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM6
        self.fc6 = nn.Linear(input_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.fc6_mean = nn.Linear(hidden_dim, z_dim)
        self.fc6_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM7
        self.fc7 = nn.Linear(input_dim, hidden_dim)
        self.bn7 = nn.BatchNorm1d(hidden_dim)
        self.fc7_mean = nn.Linear(hidden_dim, z_dim)
        self.fc7_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM8
        self.fc8 = nn.Linear(input_dim, hidden_dim)
        self.bn8 = nn.BatchNorm1d(hidden_dim)
        self.fc8_mean = nn.Linear(hidden_dim, z_dim)
        self.fc8_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM9
        self.fc9 = nn.Linear(input_dim, hidden_dim)
        self.bn9 = nn.BatchNorm1d(hidden_dim)
        self.fc9_mean = nn.Linear(hidden_dim, z_dim)
        self.fc9_logvar = nn.Linear(hidden_dim, z_dim)

        self.gmm_weights = nn.Parameter(torch.ones(9) / 2, requires_grad=True)

        
        # Add some validation on gmm_weights
        if self.gmm_weights.min() < 0:
            self.gmm_weights.clamp_(min=0)

        if abs(self.gmm_weights.sum() - 1) > 1e-3:
            self.gmm_weights = nn.Parameter(self.gmm_weights / self.gmm_weights.sum())

        
        self.fc_d1 = nn.Linear(z_dim, hidden_dim)
        self.bn_d1 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d1 = nn.Dropout(dropout_rate)

        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_d2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d2 = nn.Dropout(dropout_rate)

        self.fc_d3 = nn.Linear(hidden_dim, hidden_dim)  
        self.bn_d3 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d3 = nn.Dropout(dropout_rate)

        """
        fc_count: Maps the hidden representation to the predicted total count for each input dimension size
        fc_probs: Maps the hidden representation to the predicted probability of zero inflation for eachinput dimension size
        fc_logits: Maps the hidden representation to the predicted logits (for the negative binomial dispersion) for each input dimension size
        """

        self.fc_count = nn.Linear(hidden_dim, input_dim)  # total_count output
        self.fc_probs = nn.Linear(hidden_dim, input_dim)  # probs output
        self.fc_logits = nn.Linear(hidden_dim, input_dim)  # logits output

        self.z_dim = z_dim

        self.input_dim = input_dim

    def encode(self, x):
        h1 = nn.ReLU()(self.bn1(self.fc1(x)))
        h1 = self.dropout(h1)
        h2 = nn.ReLU()(self.bn2(self.fc2(x)))
        h2 = self.dropout(h2)
        h3 = nn.ReLU()(self.bn3(self.fc3(x)))
        h3 = self.dropout(h3)
        h4 = nn.ReLU()(self.bn4(self.fc4(x)))
        h4 = self.dropout(h4)
        h5 = nn.ReLU()(self.bn5(self.fc5(x)))
        h5 = self.dropout(h5)
        h6 = nn.ReLU()(self.bn6(self.fc6(x)))
        h6 = self.dropout(h6)
        h7 = nn.ReLU()(self.bn7(self.fc7(x)))
        h7 = self.dropout(h7)
        h8 = nn.ReLU()(self.bn8(self.fc8(x)))
        h8 = self.dropout(h8)
        h9 = nn.ReLU()(self.bn9(self.fc9(x)))
        h9 = self.dropout(h9)

        mu1 = self.fc1_mean(h1)
        logvar1 = self.fc1_logvar(h1)
        mu2 = self.fc2_mean(h2)
        logvar2 = self.fc2_logvar(h2)
        mu3 = self.fc3_mean(h3)
        logvar3 = self.fc3_logvar(h3)
        mu4 = self.fc4_mean(h4)
        logvar4 = self.fc4_logvar(h4)
        mu5 = self.fc5_mean(h5)
        logvar5 = self.fc5_logvar(h5)
        mu6 = self.fc6_mean(h6)
        logvar6 = self.fc6_logvar(h6)
        mu7 = self.fc7_mean(h7)
        logvar7 = self.fc7_logvar(h7)
        mu8 = self.fc8_mean(h8)
        logvar8 = self.fc8_logvar(h8)
        mu9 = self.fc9_mean(h9)
        logvar9 = self.fc9_logvar(h9)
        
        var1 = F.softplus(logvar1)
        logvar1 = torch.log(var1 + 1e-8)
        var2 = F.softplus(logvar2)
        logvar2 = torch.log(var2 + 1e-8)
        var3 = F.softplus(logvar3)
        logvar3 = torch.log(var3 + 1e-8)
        var4 = F.softplus(logvar4)
        logvar4 = torch.log(var4 + 1e-8)
        var5 = F.softplus(logvar5)
        logvar5 = torch.log(var5 + 1e-8)
        var6 = F.softplus(logvar6)
        logvar6 = torch.log(var6 + 1e-8)
        var7 = F.softplus(logvar7)
        logvar7 = torch.log(var7 + 1e-8)
        var8 = F.softplus(logvar8)
        logvar8 = torch.log(var8 + 1e-8)
        var9 = F.softplus(logvar9)
        logvar9 = torch.log(var9 + 1e-8)

        return mu1, logvar1, mu2, logvar2, mu3, logvar3, mu4, logvar4, mu5, logvar5, mu6, logvar6, mu7, logvar7, mu8, logvar8, mu9, logvar9


    def reparameterize(self, mu1, logvar1, mu2, logvar2, mu3, logvar3, mu4, logvar4, mu5, logvar5, mu6, logvar6, mu7, logvar7, mu8, logvar8, mu9, logvar9):
        # Calculate std from logvar
        std1 = torch.exp(0.1 * logvar1)
        std2 = torch.exp(0.1 * logvar2) 
        std3 = torch.exp(0.1 * logvar3)
        std4 = torch.exp(0.1 * logvar4)
        std5 = torch.exp(0.1 * logvar5)
        std6 = torch.exp(0.1 * logvar6)
        std7 = torch.exp(0.1 * logvar7)
        std8 = torch.exp(0.1 * logvar8)
        std9 = torch.exp(0.1 * logvar9)

        # Sample epsilon
        eps1 = torch.randn_like(std1)
        eps2 = torch.randn_like(std2)
        eps3 = torch.randn_like(std3) 
        eps4 = torch.randn_like(std4)
        eps5 = torch.randn_like(std5)
        eps6 = torch.randn_like(std6)
        eps7 = torch.randn_like(std7)
        eps8 = torch.randn_like(std8)
        eps9 = torch.randn_like(std9)

        # Reparameterize each component
        z1 = mu1 + eps1 * std1
        z2 = mu2 + eps2 * std2
        z3 = mu3 + eps3 * std3
        z4 = mu4 + eps4 * std4
        z5 = mu5 + eps5 * std5
        z6 = mu6 + eps6 * std6
        z7 = mu7 + eps7 * std7
        z8 = mu8 + eps8 * std8
        z9 = mu9 + eps9 * std9

        # Create all options 
        z_stack = torch.stack([z1, z2, z3, z4, z5, z6, z7, z8, z9], dim=-1)

        # Sample GMM components per sample with Gumbel-Softmax trick
        gmm_weights = self.gmm_weights.unsqueeze(0).expand(z_stack.shape[0], -1)  # Make sure weights are the same size as the batch
        gmm_weights = F.gumbel_softmax(gmm_weights, tau=1, hard=False, dim=-1)  # Using Gumbel-Softmax relaxation
        gmm_weights = gmm_weights.unsqueeze(-1)  # Add extra dimension for multiplication

        gmm_weights = gmm_weights.transpose(-1, -2)  # gmm_weights is now of shape [30, 1, 6]
        gmm_weights = gmm_weights.expand(-1, self.z_dim, -1)  # gmm_weights is now of shape [30, 20, 6]

        z = (z_stack * gmm_weights).sum(dim=-1)  # the multiplication and sum should now work correctly

        return z


    def decode(self, z):
        h3 = nn.ReLU()(self.bn_d1(self.fc_d1(z)))
        h3 = self.dropout_d1(h3)

        h4 = nn.ReLU()(self.bn_d2(self.fc_d2(h3)))
        h4 = self.dropout_d2(h4)

        h5 = nn.ReLU()(self.bn_d3(self.fc_d3(h4)))
        h5 = self.dropout_d3(h5)

        return nn.ReLU()(self.fc_count(h5)), torch.clamp(nn.Sigmoid()(self.fc_probs(h5)), min=1e-4, max=1-1e-4), self.fc_logits(h5)

    def forward(self, x):
        mu1, logvar1, mu2, logvar2, mu3, logvar3, mu4, logvar4, mu5, logvar5, mu6, logvar6, mu7, logvar7, mu8, logvar8, mu9, logvar9 = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu1, logvar1, mu2, logvar2, mu3, logvar3, mu4, logvar4, mu5, logvar5, mu6, logvar6, mu7, logvar7, mu8, logvar8, mu9, logvar9)
        return self.decode(z), mu1, logvar1, mu2, logvar2, mu3, logvar3, mu4, logvar4, mu5, logvar5, mu6, logvar6, mu7, logvar7, mu8, logvar8, mu9, logvar9



import torch.nn as nn

class bulkVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout_rate=0.1):
        super(bulkVAE, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # GMM1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1_mean = nn.Linear(hidden_dim, z_dim)
        self.fc1_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM2
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, z_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM3
        self.fc3 = nn.Linear(input_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, z_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM4
        self.fc4 = nn.Linear(input_dim, hidden_dim)        
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc4_mean = nn.Linear(hidden_dim, z_dim)
        self.fc4_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM5
        self.fc5 = nn.Linear(input_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.fc5_mean = nn.Linear(hidden_dim, z_dim)
        self.fc5_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM6
        self.fc6 = nn.Linear(input_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.fc6_mean = nn.Linear(hidden_dim, z_dim)
        self.fc6_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM7
        self.fc7 = nn.Linear(input_dim, hidden_dim)
        self.bn7 = nn.BatchNorm1d(hidden_dim)
        self.fc7_mean = nn.Linear(hidden_dim, z_dim)
        self.fc7_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM8
        self.fc8 = nn.Linear(input_dim, hidden_dim)
        self.bn8 = nn.BatchNorm1d(hidden_dim)
        self.fc8_mean = nn.Linear(hidden_dim, z_dim)
        self.fc8_logvar = nn.Linear(hidden_dim, z_dim)
        # GMM9
        self.fc9 = nn.Linear(input_dim, hidden_dim)
        self.bn9 = nn.BatchNorm1d(hidden_dim)
        self.fc9_mean = nn.Linear(hidden_dim, z_dim)
        self.fc9_logvar = nn.Linear(hidden_dim, z_dim)

        self.gmm_weights = nn.Parameter(torch.ones(9) / 2, requires_grad=True)

        # Add some validation on gmm_weights
        if self.gmm_weights.min() < 0:
            self.gmm_weights.clamp_(min=0)

        if abs(self.gmm_weights.sum() - 1) > 1e-3:
            self.gmm_weights = nn.Parameter(self.gmm_weights / self.gmm_weights.sum())
        
        self.fc_d1 = nn.Linear(z_dim, hidden_dim)
        self.bn_d1 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d1 = nn.Dropout(dropout_rate)

        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_d2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d2 = nn.Dropout(dropout_rate)

        self.fc_d3 = nn.Linear(hidden_dim, input_dim)

        # self.fc_count = nn.Linear(z_dim, input_dim)  
        # self.fc_probs = nn.Linear(z_dim, input_dim)   
        # self.fc_logits = nn.Linear(z_dim, input_dim)    

        self.z_dim = z_dim

        self.input_dim = input_dim

    def encode(self, x):
        h1 = nn.ReLU()(self.bn1(self.fc1(x)))
        h1 = self.dropout(h1)
        h2 = nn.ReLU()(self.bn2(self.fc2(x)))
        h2 = self.dropout(h2)
        h3 = nn.ReLU()(self.bn3(self.fc3(x)))
        h3 = self.dropout(h3)
        h4 = nn.ReLU()(self.bn4(self.fc4(x)))
        h4 = self.dropout(h4)
        h5 = nn.ReLU()(self.bn5(self.fc5(x)))
        h5 = self.dropout(h5)
        h6 = nn.ReLU()(self.bn6(self.fc6(x)))
        h6 = self.dropout(h6)
        h7 = nn.ReLU()(self.bn7(self.fc7(x)))
        h7 = self.dropout(h7)
        h8 = nn.ReLU()(self.bn8(self.fc8(x)))
        h8 = self.dropout(h8)
        h9 = nn.ReLU()(self.bn9(self.fc9(x)))
        h9 = self.dropout(h9)

        m1 = self.fc1_mean(h1)
        logvar1 = self.fc1_logvar(h1)
        m2 = self.fc2_mean(h2)
        logvar2 = self.fc2_logvar(h2)
        m3 = self.fc3_mean(h3)
        logvar3 = self.fc3_logvar(h3)
        m4 = self.fc4_mean(h4)
        logvar4 = self.fc4_logvar(h4)
        m5 = self.fc5_mean(h5)
        logvar5 = self.fc5_logvar(h5)
        m6 = self.fc6_mean(h6)
        logvar6 = self.fc6_logvar(h6)
        m7 = self.fc7_mean(h7)
        logvar7 = self.fc7_logvar(h7)
        m8 = self.fc8_mean(h8)
        logvar8 = self.fc8_logvar(h8)
        m9 = self.fc9_mean(h9)
        logvar9 = self.fc9_logvar(h9)
        mus = [m1, m2, m3, m4, m5, m6, m7, m8, m9]
        logvars = [logvar1, logvar2, logvar3, logvar4, logvar5, logvar6, logvar7, logvar8, logvar9]

        return mus, logvars

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    # def decode(self, z):
    #     z = z.reshape(-1,9)

    #     h1 = nn.ReLU()(self.bn_d1(self.fc_d1(z)))
    #     h1 = self.dropout_d1(h1)

    #     h2 = nn.ReLU()(self.bn_d2(self.fc_d2(h1))) 
    #     h2 = self.dropout_d2(h2)

    #     recon_x = self.fc_d3(h2)
    #     recon_x = recon_x.sum(dim=0)
    #     return recon_x

    def forward(self, x):
        mus, logvars = self.encode(x.view(-1, self.input_dim))
        return mus, logvars
        
        # zs = [self.reparameterize(mu, logvar) for mu, logvar in zip(mus, logvars)]

        # z = sum(self.gmm_weights[i] * zs[i].view(1, -1) for i in range(9))/9

        # recon_x = self.decode(z)

        # return recon_x, mus, logvars
    

class B2SC(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout_rate=0.1):
        super(B2SC, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1_mean = nn.Linear(hidden_dim, z_dim)
        self.fc1_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, z_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc3 = nn.Linear(input_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, z_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)        
        self.fc4_mean = nn.Linear(hidden_dim, z_dim)
        self.fc4_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc5 = nn.Linear(input_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.fc5_mean = nn.Linear(hidden_dim, z_dim)
        self.fc5_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc6 = nn.Linear(input_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.fc6_mean = nn.Linear(hidden_dim, z_dim)
        self.fc6_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc7 = nn.Linear(input_dim, hidden_dim)
        self.bn7 = nn.BatchNorm1d(hidden_dim)
        self.fc7_mean = nn.Linear(hidden_dim, z_dim)
        self.fc7_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc8 = nn.Linear(input_dim, hidden_dim)
        self.bn8 = nn.BatchNorm1d(hidden_dim)
        self.fc8_mean = nn.Linear(hidden_dim, z_dim)
        self.fc8_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.fc9 = nn.Linear(input_dim, hidden_dim)
        self.bn9 = nn.BatchNorm1d(hidden_dim)
        self.fc9_mean = nn.Linear(hidden_dim, z_dim)
        self.fc9_logvar = nn.Linear(hidden_dim, z_dim)
        
        self.gmm_weights = nn.Parameter(torch.ones(9) / 9)
        
        # Decoder
        self.fc_d1 = nn.Linear(z_dim, hidden_dim)
        self.bn_d1 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d1 = nn.Dropout(dropout_rate)
        
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_d2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d2 = nn.Dropout(dropout_rate)
        
        self.fc_d3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_d3 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d3 = nn.Dropout(dropout_rate)

        self.fc_count = nn.Linear(hidden_dim, input_dim)
        # self.fc_probs = nn.Linear(hidden_dim, input_dim)  
        # self.fc_logits = nn.Linear(hidden_dim, input_dim)

        self.z_dim = z_dim
        self.input_dim = input_dim

        
    def encode(self, x):
        
        h1 = nn.ReLU()(self.bn1(self.fc1(x)))
        h1 = self.dropout(h1)
        
        h2 = nn.ReLU()(self.bn2(self.fc2(x)))
        h2 = self.dropout(h2)
        
        h3 = nn.ReLU()(self.bn3(self.fc3(x)))
        h3 = self.dropout(h3)
        
        h4 = nn.ReLU()(self.bn4(self.fc4(x)))
        h4 = self.dropout(h4)
        
        h5 = nn.ReLU()(self.bn5(self.fc5(x)))
        h5 = self.dropout(h5)
        
        h6 = nn.ReLU()(self.bn6(self.fc6(x)))
        h6 = self.dropout(h6)
        
        h7 = nn.ReLU()(self.bn7(self.fc7(x)))
        h7 = self.dropout(h7)
        
        h8 = nn.ReLU()(self.bn8(self.fc8(x)))
        h8 = self.dropout(h8)
        
        h9 = nn.ReLU()(self.bn9(self.fc9(x)))
        h9 = self.dropout(h9)
        
        m1 = self.fc1_mean(h1)
        logvar1 = self.fc1_logvar(h1)
        m2 = self.fc2_mean(h2)
        logvar2 = self.fc2_logvar(h2)
        m3 = self.fc3_mean(h3)
        logvar3 = self.fc3_logvar(h3)
        m4 = self.fc4_mean(h4)
        logvar4 = self.fc4_logvar(h4)
        m5 = self.fc5_mean(h5)
        logvar5 = self.fc5_logvar(h5)
        m6 = self.fc6_mean(h6)
        logvar6 = self.fc6_logvar(h6)
        m7 = self.fc7_mean(h7)
        logvar7 = self.fc7_logvar(h7)
        m8 = self.fc8_mean(h8)
        logvar8 = self.fc8_logvar(h8)
        m9 = self.fc9_mean(h9)
        logvar9 = self.fc9_logvar(h9)
        mus = [m1, m2, m3, m4, m5, m6, m7, m8, m9]
        logvars = [logvar1, logvar2, logvar3, logvar4, logvar5, logvar6, logvar7, logvar8, logvar9]

        return mus, logvars

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def decode(self, z):
        
        h1 = nn.ReLU()(self.bn_d1(self.fc_d1(z)))
        h1 = self.dropout_d1(h1)

        h2 = nn.ReLU()(self.bn_d2(self.fc_d2(h1)))
        h2 = self.dropout_d2(h2)

        h3 = nn.ReLU()(self.bn_d3(self.fc_d3(h2)))
        h3 = self.dropout_d3(h3)
        
        recon_x = self.fc_count(h3)
        # probs = torch.clamp(nn.Sigmoid()(self.fc_probs(h3)), 1e-4, 1-1e-4)
        # logits = self.fc_logits(h3)
        
        return recon_x#, probs, logits

    
    def forward(self, x, selected_neuron):
        # pdb.set_trace()
        
        mus, logvars = self.encode(x)
        
        z = self.reparameterize(mus[selected_neuron], logvars[selected_neuron])
        
        recon_x = self.decode(z)
        
        return recon_x.sum(dim=0)#, probs, logits
    

