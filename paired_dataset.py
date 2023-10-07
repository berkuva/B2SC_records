import scanpy as sc
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime

import matplotlib.patches as mpatches

seed = 200
np.random.seed(seed)
torch.manual_seed(seed)

data_dir = "/u/hc2kc/scVAE/pbmc1k/data/"
# data_dir = "/home/hc2kc/Desktop/raw_feature_bc_matrix/"
adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
# adata = sc.read_h5ad(data_dir+"pbmc10k.h5ad")
# import pdb;pdb.set_trace()
barcode_path = data_dir+'barcode_to_celltype.csv'
barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
barcodes_with_labels.columns = ['barcodes', 'labels']



# Step 2: Clean the labels data
# Remove rows with 'unknown' or NaN labels
barcodes_with_labels = barcodes_with_labels[
    # (barcodes_with_labels['labels'].notna()) &
    (barcodes_with_labels['labels'] != 'Unknown')
]

# Here's the cleaned labels after filtering
cleaned_labels = barcodes_with_labels['labels'].values

labels = cleaned_labels

# Step 3: Filter the adata object
# Retain only the observations that match the filtered barcodes
filtered_barcodes = barcodes_with_labels['barcodes'].values
adata = adata[adata.obs.index.isin(filtered_barcodes)]

# Display the filtered AnnData object
print(np.unique(labels))

assert len(labels) == adata.n_obs

adata.obs['labels'] = labels
# Make sure barcodes in adata are in the same format as barcodes_with_labels, if not, format them.
adata.obs['barcodes'] = adata.obs.index

adata.X = adata.X.toarray()
adata.obs.index = adata.obs.index.astype(str)

mapping_dict = \
            {
            'CD8+ NKT-like cells': 0,
            'Classical Monocytes': 1,
            'Memory CD4+ T cells': 2,
            'Naive B cells': 3,
            'Naive CD8+ T cells': 4,
            'Natural killer  cells': 5,
            'Non-classical monocytes': 6,
            'Plasmacytoid Dendritic cells': 7,
            'Platelets': 8,
            'Pre-B cells': 9
            }

# Apply mapping to the 'labels' column of adata.obs
# import pdb;pdb.set_trace()
adata.obs['labels'] = adata.obs['labels'].astype('category')
adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)


# Continue with your existing code
# labels = adata.obs['labels'].cat.codes.values
# import pdb;pdb.set_trace()
cell_types_tensor = torch.LongTensor(np.copy(adata.obs['labels']))

# Randomly select  cells from your AnnData object
how_many = 1129
indices = np.random.choice(len(adata), how_many, replace=False)
print(indices[:10])

# unselected = np.setdiff1d(np.arange(len(adata)), indices)
# print(unselected[:10])

# Create a TensorDataset from your AnnData object
X_tensor = torch.Tensor(adata.X[indices])#
cell_types_tensor = cell_types_tensor[indices]#


# Save X_tensor and cell_types_tensor as recon_counts_u.npy and labels_u.npy.
# np.save('recon_counts_u.npy', X_tensor)
# np.save('labels_u.npy', cell_types_tensor)

dataset = TensorDataset(X_tensor, cell_types_tensor)
mini_batch = how_many

dataloader = DataLoader(dataset, batch_size=mini_batch, shuffle=False)
# import pdb;pdb.set_trace()
input_dim = 33538
hidden_dim = 700
z_dim = 10

__all__ = ["mapping_dict", "dataloader", 'X_tensor', "cell_types_tensor", "mini_batch"]
