import scanpy as sc
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import collections

# np.random.seed(1)
# torch.manual_seed(1)


data_dir = "B2SC/filtered_gene_bc_matrices/hg19/"
#data_dir = "/u/hc2kc/hg19/"
# data_dir = "/home/hc2kc/Rivanna/scVAE/hg19/"

adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

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
    'Naive B cells': '0',
    'Non-classical monocytes': '1',
    'Classical Monocytes': '2',
    'Natural killer  cells': '3',
    'CD8+ NKT-like cells': '4',
    'Memory CD4+ T cells': '5',
    'Naive CD8+ T cells': '6',
    'Myeloid Dendritic cells': '7',
    'Platelets': '8'
}

# Apply mapping to the 'labels' column of adata.obs
adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)

adata.obs['labels'] = adata.obs['labels'].astype('category')
labels = adata.obs['labels'].cat.codes.values
labels = torch.LongTensor(labels)




# 419

# Create a dictionary of labels and their corresponding indices
label_indices = collections.defaultdict(list)
for idx, label in enumerate(labels):
    label_indices[label.item()].append(idx)

# Create a TensorDataset from your AnnData object
tensor_x = torch.Tensor(adata.X)
dataset = TensorDataset(tensor_x, labels)
# pdb.set_trace()

mini_batch = 1000
# Dataloader
loader = DataLoader(dataset, batch_size=mini_batch, shuffle=False)


__all__ = ['adata', 'loader']
