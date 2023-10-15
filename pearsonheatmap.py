import scanpy as sc
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import collections
import anndata as ad

#Load data
np.random.seed(4)

data_dir = "B2SC/filtered_gene_bc_matrices/hg19/"

adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

gendata = np.load('B2SC/filtered_gene_bc_matrices/hg19/recon_counts.npy')
genlabels = np.load('B2SC/filtered_gene_bc_matrices/hg19/labels.npy')


gen = ad.AnnData(X = gendata)
gen.var_names = adata.var_names
adata.X = adata.X.toarray()



import seaborn as sns
import matplotlib.pyplot as plt

num_genes = 100

def select_genes(data):
    gene_mean = data.X.mean(axis=0)
    gene_std = data.X.std(axis=0)
    gene_cv = gene_std / gene_mean
    gene_cv = np.nan_to_num(gene_cv)
    top_genes_indices = np.argsort(gene_cv)[::-1][:num_genes]
    return top_genes_indices

# Make a copy of the adata object stored in adata_temp
adata_temp = adata.copy()
gen_temp = gen.copy()
top_variable_genes = select_genes(adata_temp)

# Subset your data to include only these genes
adata_top = adata.X[:, top_variable_genes]
# Subset gen using the same genes
gen_top = gen.X[:, top_variable_genes]

# Gen map
pearson_corr = np.corrcoef(gen_top, rowvar=False)

# Create heatmap
a = sns.heatmap(pearson_corr, cmap="coolwarm", vmin = -1, vmax = 1)

# Save the figure
a.set_title('Pearson Correlation Coefficient Heatmap')
a.set_xlabel('Generated Data')
a.set_ylabel('Generated Data')
a.set_xlim(0, 100)
a.set_ylim(0, 100)
a.invert_yaxis()

plt.savefig('B2SC/figures/heatmap_generated.png')
plt.clf()

# Real Map
pearson_corr = np.corrcoef(adata_top, rowvar=False)

# Create heatmap
a = sns.heatmap(pearson_corr, cmap="coolwarm", vmin = -1, vmax = 1)

# Save the figure
a.set_title('Pearson Correlation Coefficient Heatmap')
a.set_xlabel('Real Data')
a.set_ylabel('Real Data')
a.set_xlim(0, 100)
a.set_ylim(0, 100)
a.invert_yaxis()

plt.savefig('B2SC/figures/heatmap_control.png')
plt.clf()
