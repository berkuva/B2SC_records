import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu" 
print(device)


class scVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, gmm_num=9, dropout_rate=0.1):
        super(scVAE, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        
        # Generalize for gmm_num mixture models
        self.fcs = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(gmm_num)])
        self.fcs_mean = nn.ModuleList([nn.Linear(hidden_dim, z_dim) for _ in range(gmm_num)])
        self.fcs_logvar = nn.ModuleList([nn.Linear(hidden_dim, z_dim) for _ in range(gmm_num)])

        # Linear layer to produce gmm_weights
        self.fc_gmm_weights = nn.Linear(input_dim, gmm_num)

        self.fc_d1 = nn.Linear(z_dim, hidden_dim)
        self.bn_d1 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d1 = nn.Dropout(dropout_rate)

        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_d2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d2 = nn.Dropout(dropout_rate)

        self.fc_d3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_d3 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d3 = nn.Dropout(dropout_rate)

        self.fc_mean = nn.Linear(hidden_dim, input_dim)
        self.fc_zero_inflation = nn.Linear(hidden_dim, input_dim)
        self.theta = nn.Parameter(torch.ones(input_dim) * 0.5)

        self.z_dim = z_dim
        self.input_dim = input_dim
        self.gmm_num = gmm_num

    def encode(self, x):
        mus = []
        logvars = []

        for i in range(self.gmm_num):
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

        for i in range(self.gmm_num):
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
    def __init__(self, input_dim, hidden_dim, z_dim, num_gmms=9, dropout_rate=0.1):
        super(bulkVAE, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.num_gmms = num_gmms

        # GMMs
        self.fcs = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_gmms)])
        # self.bns = nn.ModuleList([nn.InstanceNorm1d(hidden_dim) for _ in range(num_gmms)])
        self.fc_means = nn.ModuleList([nn.Linear(hidden_dim, z_dim) for _ in range(num_gmms)])
        self.fc_logvars = nn.ModuleList([nn.Linear(hidden_dim, z_dim) for _ in range(num_gmms)])

        # Linear layer to produce gmm_weights
        self.fc_gmm_weights = nn.Linear(input_dim, num_gmms)

        self.fc_d1 = nn.Linear(z_dim, hidden_dim)
        # self.bn_d1 = nn.InstanceNorm1d(hidden_dim)
        self.dropout_d1 = nn.Dropout(dropout_rate)

        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn_d2 = nn.InstanceNorm1d(hidden_dim)
        self.dropout_d2 = nn.Dropout(dropout_rate)

        self.fc_d3 = nn.Linear(hidden_dim, input_dim)

        self.z_dim = z_dim
        self.input_dim = input_dim

    def encode(self, x):
        mus = []
        logvars = []

        for i in range(self.num_gmms):
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
    def __init__(self, input_dim, hidden_dim, z_dim, num_gmms=9, dropout_rate=0.1):
        super(B2SC, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.num_gmms = num_gmms

        # Encoder
        self.fcs = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_gmms)])
        self.fc_means = nn.ModuleList([nn.Linear(hidden_dim, z_dim) for _ in range(num_gmms)])
        self.fc_logvars = nn.ModuleList([nn.Linear(hidden_dim, z_dim) for _ in range(num_gmms)])

        # Linear layer to produce gmm_weights
        self.fc_gmm_weights = nn.Linear(input_dim, num_gmms)
        
        # Decoder
        self.fc_d1 = nn.Linear(z_dim, hidden_dim)
        # self.bn_d1 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d1 = nn.Dropout(dropout_rate)

        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn_d2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d2 = nn.Dropout(dropout_rate)

        self.fc_d3 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn_d3 = nn.BatchNorm1d(hidden_dim)
        self.dropout_d3 = nn.Dropout(dropout_rate)

        self.fc_count = nn.Linear(hidden_dim, input_dim)

        self.z_dim = z_dim
        self.input_dim = input_dim

    def encode(self, x):
        mus = []
        logvars = []

        for i in range(self.num_gmms):
            h = nn.ReLU()(self.fcs[i](x))
            h = self.dropout(h)
            mus.append(self.fc_means[i](h))
            logvars.append(self.fc_logvars[i](h))
        
        # Produce gmm_weights using the linear layer and softmax activation
        gmm_weights = self.fc_gmm_weights(x)

        return mus, logvars, gmm_weights

    def reparameterize(self, mus, logvars, gmm_weights, selected_neuron):
        zs = []
        mu = mus[selected_neuron]
        std = torch.exp(0.1 * logvars[selected_neuron])
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

    def forward(self, x, selected_neuron):
        mus, logvars, gmm_weights = self.encode(x)
        z = self.reparameterize(mus, logvars, gmm_weights, selected_neuron)
        recon_x = self.decode(z)
        
        return recon_x.sum(dim=0)
