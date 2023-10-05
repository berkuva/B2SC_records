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

# Create a pandas Series from genlabels
genlabels_series = pd.Series(genlabels)

# Read your barcodes_with_labels file
barcode_path = 'B2SC/filtered_gene_bc_matrices/hg19/barcodes_with_labels.txt'
#barcode_path = '/u/hc2kc/hg19/barcodes_with_labels.txt'
# barcode_path = "/home/hc2kc/Rivanna/scVAE/hg19/barcodes_with_labels.txt"
barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None)

# Assuming that the barcodes are in the first column (0) and labels in the second (1)
barcodes_with_labels.columns = ['barcodes', 'labels']

# Make sure barcodes in adata are in the same format as barcodes_with_labels, if not, format them.
adata.obs['barcodes'] = adata.obs.index

# Merge adata.obs and barcodes_with_labels on barcodes
adata.obs = adata.obs.reset_index(drop=True)
adata.obs = adata.obs.merge(barcodes_with_labels, on='barcodes', how='left')



# Now your labels are part of the adata object and can be accessed using adata.obs['labels']
# Convert to dense array if necessary
adata.X = adata.X.toarray()
adata.obs.index = adata.obs.index.astype(str)

# sc.pp.filter_genes_dispersion(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.filter_cells(adata, min_genes=10)

# Calculate standard deviation for each gene across cells
gene_std_devs = np.std(adata.X, axis=0)

# Select top variable genes based on standard deviation
top_variable_genes = np.argsort(gene_std_devs)

# Subset your data to include only these genes
adata = adata[:, top_variable_genes]
# Doublet-removed adata
# adata = remove_doublets(adata)
# Your mapping dictionary
mapping_dict = {
    'Naive B cells': '1',
    'Non-classical monocytes': '1',
    'Classical Monocytes': '1',
    'Natural killer  cells': '1',
    'CD8+ NKT-like cells': '1',
    'Memory CD4+ T cells': '1',
    'Naive CD8+ T cells': '1',
    'Myeloid Dendritic cells': '1',
    'Platelets': '1'
}

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
        adata.obs.at[str(random_row_indices[i]),'labels'] = '2'
    return dataset

# Apply mapping to the 'labels' column of adata.obs
adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)

gen = ad.AnnData(X = gendata)
gen.var_names = adata.var_names

adata = ad.concat([adata, gen], join="outer")
labels_list = ['1'] * 2700 + ['0'] * 2700

# Set the 'labels' column of the adata.obs DataFrame to the shuffled list of labels
adata.obs['labels'] = labels_list
apply_gaussian_noise(adata.X, 0, 0.2, 1350)

adata.obs['labels'] = adata.obs['labels'].astype('category')

labels = adata.obs['labels'].cat.codes.values

# Shuffle the indices
shuffled_indices = np.random.permutation(len(adata.X))

# Rearrange both arrays using the shuffled indices
adata.X = adata.X[shuffled_indices]
labels = labels[shuffled_indices]


labels = torch.LongTensor(labels)


# Create a dictionary of labels and their corresponding indices
label_indices = collections.defaultdict(list)
for idx, label in enumerate(labels):
    label_indices[label.item()].append(idx)
# Create a TensorDataset from your AnnData object
tensor_x = torch.Tensor(adata.X)

dataset = TensorDataset(tensor_x, labels)
# pdb.set_trace()

# Dataloader
loader = DataLoader(dataset, batch_size=5400, shuffle=False)


__all__ = ['adata', 'loader']
