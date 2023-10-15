import scanpy as sc
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import collections
import anndata as ad

np.random.seed(4)

data_dir = "B2SC/filtered_gene_bc_matrices/hg19/"

adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

gendata = np.load('B2SC/filtered_gene_bc_matrices/hg19/recon_counts.npy')
genlabels = np.load('B2SC/filtered_gene_bc_matrices/hg19/labels.npy')


gen = ad.AnnData(X = gendata)
gen.var_names = adata.var_names
adata.X = adata.X.toarray()

# Select top 100 genes based on std/mean
mean = np.mean(adata.X, axis=0)
std = np.std(adata.X, axis=0)
std_mean_ratio = std / mean
top_100_genes = np.argsort(std_mean_ratio)[::-1][:100]
adata_temp = adata.X[:, top_100_genes]

print(adata_temp)

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate Pearson correlation coefficients
corr_matrix = np.corrcoef(adata.X.T)

# Create heatmap
a = sns.heatmap(corr_matrix, cmap='coolwarm')
a.set_title('Pearson Correlation Coefficient Heatmap')
a.set_xlabel('Real Data')
a.set_ylabel('Real Data')
a.set_xlim(0, 100)
a.set_ylim(0, 100)
a.invert_yaxis()

plt.savefig('B2SC/figures/pheat.png')
plt.clf()