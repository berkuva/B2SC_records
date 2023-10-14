import torch
from torch import nn
from torch.nn import functional as F


class scVAE(nn.Module):
    def __init__(self, args):
        super(scVAE, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        
        # Generalize for z_dim mixture models
        self.fcs = nn.ModuleList([nn.Linear(args.input_dim, args.hidden_dim) for _ in range(args.z_dim)])
        self.fcs_mean = nn.ModuleList([nn.Linear(args.hidden_dim, args.z_dim) for _ in range(args.z_dim)])
        self.fcs_logvar = nn.ModuleList([nn.Linear(args.hidden_dim, args.z_dim) for _ in range(args.z_dim)])

        # Linear layer to produce gmm_weights
        self.fc_gmm_weights = nn.Linear(args.input_dim, args.z_dim)

        self.fc_d1 = nn.Linear(args.z_dim, args.hidden_dim)
        self.bn_d1 = nn.BatchNorm1d(args.hidden_dim)
        self.dropout_d1 = nn.Dropout(args.dropout)

        self.fc_d2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.bn_d2 = nn.BatchNorm1d(args.hidden_dim)
        self.dropout_d2 = nn.Dropout(args.dropout)

        self.fc_d3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.bn_d3 = nn.BatchNorm1d(args.hidden_dim)
        self.dropout_d3 = nn.Dropout(args.dropout)

        self.fc_mean = nn.Linear(args.hidden_dim, args.input_dim)
        self.fc_zero_inflation = nn.Linear(args.hidden_dim, args.input_dim)
        self.theta = nn.Parameter(torch.ones(args.input_dim) * 0.5)

        self.z_dim = args.z_dim
        self.input_dim = args.input_dim

    def encode(self, x):
        mus = []
        logvars = []

        for i in range(self.z_dim):
            h = nn.ReLU()(self.fcs[i](x))
            h = self.dropout(h)
            mus.append(self.fcs_mean[i](h))
            logvar = self.fcs_logvar[i](h)
            var = F.softplus(logvar)
            logvars.append(torch.log(var + 1e-8))

        gmm_weights = self.fc_gmm_weights(x)

        return mus, logvars, gmm_weights

    def reparameterize(self, mus, logvars, gmm_weights):
        zs = []

        for i in range(self.z_dim):
            std = torch.exp(0.1 * logvars[i])
            eps = torch.randn_like(std)
            zs.append(mus[i] + eps * std)

        z_stack = torch.stack(zs, dim=-1)
        gmm_weights = gmm_weights.unsqueeze(-1).transpose(-1, -2).expand(-1, self.z_dim, -1)
        z = (z_stack * gmm_weights).sum(dim=-1)

        return z

    def decode(self, z):
        h3 = nn.ReLU()(self.fc_d1(z))
        h3 = self.dropout_d1(h3)

        h4 = nn.ReLU()(self.fc_d2(h3))
        h4 = self.dropout_d2(h4)

        h5 = nn.ReLU()(self.fc_d3(h4))
        h5 = self.dropout_d3(h5)

        preds = nn.ReLU()(self.fc_mean(h5))
        zero_inflation_prob = torch.sigmoid(self.fc_zero_inflation(h5))

        return preds, zero_inflation_prob, self.theta
    
    
    def forward(self, x):
        mus, logvars, gmm_weights = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mus, logvars, gmm_weights)
        preds, zero_inflation_prob, theta = self.decode(z)
        return preds, zero_inflation_prob, theta, mus, logvars, gmm_weights
    

class bulkVAE(nn.Module):
    def __init__(self, args):
        super(bulkVAE, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.z_dim = args.z_dim
        self.input_dim = args.input_dim

        # GMMs
        self.fcs = nn.ModuleList([nn.Linear(args.input_dim, args.hidden_dim) for _ in range(args.z_dim)])
        # self.bns = nn.ModuleList([nn.InstanceNorm1d(hidden_dim) for _ in range(args.z_dim)])
        self.fc_means = nn.ModuleList([nn.Linear(args.hidden_dim, args.z_dim) for _ in range(args.z_dim)])
        self.fc_logvars = nn.ModuleList([nn.Linear(args.hidden_dim, args.z_dim) for _ in range(args.z_dim)])

        # Linear layer to produce gmm_weights
        self.fc_gmm_weights = nn.Linear(args.input_dim, args.z_dim)

        self.fc_d1 = nn.Linear(args.z_dim, args.hidden_dim)
        self.dropout_d1 = nn.Dropout(args.dropout)

        self.fc_d2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout_d2 = nn.Dropout(args.dropout)

        self.fc_d3 = nn.Linear(args.hidden_dim, args.input_dim)

        self.z_dim = args.z_dim
        self.input_dim = args.input_dim

    def encode(self, x):
        mus = []
        logvars = []

        for i in range(self.z_dim):
            h = nn.ReLU()(self.fcs[i](x))
            h = self.dropout(h)
            mus.append(self.fc_means[i](h))
            logvars.append(self.fc_logvars[i](h))
        
        # Produce gmm_weights using the linear layer and softmax activation
        gmm_weights = self.fc_gmm_weights(x)

        return mus, logvars, gmm_weights

    def forward(self, x):
        mus, logvars, gmm_weights = self.encode(x.view(-1, self.input_dim))
        return mus, logvars, gmm_weights


class B2SC(nn.Module):
    def __init__(self, args):
        super(B2SC, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.z_dim = args.z_dim

        # Encoder
        self.fcs = nn.ModuleList([nn.Linear(args.input_dim, args.hidden_dim) for _ in range(args.z_dim)])
        self.fc_means = nn.ModuleList([nn.Linear(args.hidden_dim, args.z_dim) for _ in range(args.z_dim)])
        self.fc_logvars = nn.ModuleList([nn.Linear(args.hidden_dim, args.z_dim) for _ in range(args.z_dim)])

        # Linear layer to produce gmm_weights
        self.fc_gmm_weights = nn.Linear(args.input_dim, args.z_dim)
        
        # Decoder
        self.fc_d1 = nn.Linear(args.z_dim,args. hidden_dim)
        self.dropout_d1 = nn.Dropout(args.dropout)

        self.fc_d2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout_d2 = nn.Dropout(args.dropout)

        self.fc_d3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout_d3 = nn.Dropout(args.dropout)

        self.fc_count = nn.Linear(args.hidden_dim, args.input_dim)

        self.z_dim = args.z_dim
        self.input_dim = args.input_dim

    def encode(self, x):
        mus = []
        logvars = []

        for i in range(self.z_dim):
            h = nn.ReLU()(self.fcs[i](x))
            h = self.dropout(h)
            mus.append(self.fc_means[i](h))
            logvars.append(self.fc_logvars[i](h))
        
        # Produce gmm_weights using the linear layer and softmax activation
        gmm_weights = self.fc_gmm_weights(x)

        return mus, logvars, gmm_weights

    def reparameterize(self, mus, logvars, gmm_weights, selected_neuron, variability=1):
        zs = []
        mu = mus[selected_neuron]

        # Min-max normalize logvars to allow variance of generated cell types.
        x = logvars[selected_neuron]

        # Compute the min and max of x
        x_min = torch.min(x)
        x_max = torch.max(x)

        # Apply Min-Max scaling
        scaled_x = (x - x_min) / (x_max - x_min)
        scaled_x = variability * scaled_x #Change the integer to allow more variability.
        std = torch.exp(scaled_x)

        eps = torch.randn_like(std)

        zs.append(mu + eps * std)
        z_stack = torch.stack(zs, dim=-1)
        z = (z_stack * gmm_weights).sum(dim=-1)
        return z

    def decode(self, z):
        h1 = nn.ReLU()(self.fc_d1(z))
        h1 = self.dropout_d1(h1)

        h2 = nn.ReLU()(self.fc_d2(h1))
        h2 = self.dropout_d2(h2)

        h3 = nn.ReLU()(self.fc_d3(h2))
        h3 = self.dropout_d3(h3)
        
        recon_x = self.fc_count(h3)
        
        return recon_x

    def forward(self, x, selected_neuron=None, variability=1):
        mus, logvars, gmm_weights = self.encode(x)
        
        if selected_neuron == None:
            gmm_weights = gmm_weights.mean(0)
            # clamp the negative values to the smallest positive number
            min_positive = torch.min(gmm_weights[gmm_weights > 0]).item() if torch.any(gmm_weights > 0) else 1e-3
            gmm_weights_clamped = torch.clamp(gmm_weights, min=min_positive)
            selected_neuron = torch.multinomial(gmm_weights_clamped, 1).item()
        
        z = self.reparameterize(mus, logvars, gmm_weights, selected_neuron, variability)
        recon_x = self.decode(z)
        
        return recon_x.sum(dim=0), selected_neuron
