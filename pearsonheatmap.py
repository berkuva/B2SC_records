import scanpy as sc
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import collections
import anndata as ad

# np.random.seed(1)
# torch.manual_seed(1)

np.random.seed(4)

data_dir = "B2SC/filtered_gene_bc_matrices/hg19/"
#data_dir = "/u/hc2kc/hg19/"
# data_dir = "/home/hc2kc/Rivanna/scVAE/hg19/"

adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

gendata = np.load('B2SC/filtered_gene_bc_matrices/hg19/recon_counts.npy')
genlabels = np.load('B2SC/filtered_gene_bc_matrices/hg19/labels.npy')


gen = ad.AnnData(X = gendata)
gen.var_names = adata.var_names
adata.X = adata.X.toarray()

# sc.pp.filter_genes_dispersion(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.filter_cells(adata, min_genes=10)

def apply_gaussian_noise(dataset, mean, std_dev, num_rows_to_replace):
    num_rows = 2700
    #adata.obs['labels'] = adata.obs['labels'].cat.add_categories([0])

    # Generate Gaussian noise
    #noise = np.zeros(num_rows_to_replace)

    noise = np.random.normal(mean, std_dev, num_rows_to_replace)
    #Select random rows
    random_row_indices = np.random.choice(num_rows, num_rows_to_replace, replace=False)

    # Add the noise to the original data
    for i in range(0,num_rows_to_replace):
        current_row = dataset[random_row_indices[i]]
        current_noise = noise[i]

    
        # Add noise only to non-zero values in the row
        for j in range(len(current_row)):
            if current_row[j] != 0:
                current_row[j] += current_noise
        #adata.obs.at[str(random_row_indices[i]),'labels'] = '0'
    return dataset


# Pearson Correlation Coefficient Heatmap
# Calculate Pearson correlation coefficient
# Import seaborn library
import seaborn as sns
import matplotlib.pyplot as plt

# Select 100 top variable genes based on scanpy highly_variable_genes
from scanpy.preprocessing import highly_variable_genes

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)

# Log Transformation
adata.raw = adata
sc.pp.log1p(adata)

num_genes = 100

sc.pp.highly_variable_genes(adata, n_top_genes=num_genes)
#top_variable_genes = adata.var.highly_variable

def select_genes(adata):
    adata.var['mean'] = adata.X.mean(axis=0)
    adata.var['variance'] = adata.X.var(axis=0)
    adata.var['dispersion'] = adata.var['variance'] / adata.var['mean']
    top_genes_names = adata.var['dispersion'].nlargest(num_genes).index.tolist()
    top_variable_genes = [adata.var_names.get_loc(gene_name) for gene_name in top_genes_names]
    return top_variable_genes
top_variable_genes = select_genes(adata)


# Calculate standard deviation for each gene across cells
gene_std_devs = np.std(adata.X, axis=0)
gene_means = np.mean(adata.X, axis=0)

# Calculate coefficient of variation for each gene across cells
epsilon = 1e-9 # small constant to avoid division by zero
gene_cv = gene_std_devs / (gene_means + epsilon)

# Select 100 top variable genes based on coefficient of variation
#top_variable_genes = np.argsort(gene_cv)[::-1][0:num_genes]

# Subset your data to include only these genes
adata_top = adata.X[:, top_variable_genes]
# Subset gen using the same genes
gen_top = gen.X[:, top_variable_genes]
# Calculate Pearson correlation coefficient
pearson_corr = np.corrcoef(adata_top, gen_top, rowvar=False)

#pearson_corr = np.nan_to_num(pearson_corr)

print("Min correlation:", np.min(pearson_corr))
print("Max correlation:", np.max(pearson_corr))
# Create heatmap
a = sns.heatmap(pearson_corr, cmap="coolwarm", vmin = -1, vmax = 1)

# Save the figure
a.set_title('Pearson Correlation Coefficient Heatmap')
a.set_xlabel('Real Data')
a.set_ylabel('Generated Data')
a.set_xlim(100, 200)
a.set_ylim(0, 100)
a.invert_yaxis()

plt.savefig('B2SC/figures/heatmap_generated.png')
plt.show()
plt.clf()

'''
# Map between real and noise
adata_temp = adata.X[:, top_variable_genes]
apply_gaussian_noise(adata_temp, 0, 1, 2700)
pearson_corr = np.corrcoef(adata_temp, rowvar=False)
print("Min correlation:", np.min(pearson_corr))
print("Max correlation:", np.max(pearson_corr))
# Create heatmap with custom color map
ax = sns.heatmap(pearson_corr, cmap="coolwarm", vmin=-1, vmax=1)

# Save the figure
ax.set_title('Pearson Correlation Coefficient Heatmap')
ax.set_xlabel('Noise')
ax.set_ylabel('Noise')
ax.set_xlim(0, num_genes)
ax.set_ylim(0, num_genes)
ax.invert_yaxis()
plt.savefig('B2SC/figures/heatmap_noise.png')
plt.clf()
'''

# Map between real and noise
adata_control = adata.X[:, top_variable_genes]
pearson_corr2 = np.corrcoef(gen_top, gen_top, rowvar=False)

print("Min correlation:", np.min(pearson_corr2))
print("Max correlation:", np.max(pearson_corr2))
ax = sns.heatmap(pearson_corr2, cmap="coolwarm", vmin = -1, vmax = 1)

# Save the figure
ax.set_title('Pearson Correlation Coefficient Heatmap')
ax.set_xlabel('Real Data')
ax.set_ylabel('Real Data')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.invert_yaxis()

plt.savefig('B2SC/figures/heatmap_control.png')
plt.clf()

print(pearson_corr.shape)
print(pearson_corr2.shape)
# Pearson Corr vs Pearson Corr2
plt.scatter(pearson_corr, pearson_corr2)
plt.xlabel('Pearson Correlation Coefficient with Noise')
plt.ylabel('Pearson Correlation Coefficient with Control')
plt.savefig('B2SC/figures/pearson_corr_vs_pearson_corr2.png')
plt.clf()