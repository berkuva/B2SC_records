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

data_dir = "/u/hc2kc/scVAE/pbmc5k/data/"
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

# # Ensure the barcodes in adata are formatted as a column for merging
# adata.obs['barcodes'] = adata.obs.index

# # Merge the data based on barcodes
# merged_data = adata.obs.merge(barcodes_with_labels, on='barcodes', how='inner')

# # At this point, merged_data will only contain rows where the barcodes matched.
# # Update the adata object to reflect this:
# adata = adata[merged_data.index, :]

# If you want, you can also store the labels in the adata object:
adata.obs['labels'] = labels
# Make sure barcodes in adata are in the same format as barcodes_with_labels, if not, format them.
adata.obs['barcodes'] = adata.obs.index

# import pdb;pdb.set_trace()
# Merge adata.obs and barcodes_with_labels on barcodes
# adata.obs = adata.obs.reset_index(drop=True)
# adata.obs = adata.obs.merge(barcodes_with_labels, on='barcodes', how='left')
# adata.obs = adata.obs.drop("labels_y",1)
# adata.obs.columns = ['barcodes', 'labels']

# Now your labels are part of the adata object and can be accessed using adata.obs['labels']
# Convert to dense array if necessary
adata.X = adata.X.toarray()
adata.obs.index = adata.obs.index.astype(str)

# Calculate standard deviation for each gene across cells
# gene_std_devs = np.std(adata.X, axis=0)

# Select top variable genes based on standard deviation
# top_variable_genes = np.argsort(gene_std_devs)

# Subset your data to include only these genes
# adata = adata[:, top_variable_genes]

mapping_dict = {
    'Basophils': 0,
    'CD8+ NKT-like cells': 1,
    'Classical Monocytes': 2,
    'Erythroid-like and erythroid precursor cells': 3,
    'Memory CD4+ T cells': 4,
    'Naive B cells': 5,
    'Naive CD4+ T cells': 6,
    'Natural killer  cells': 7,
    'Non-classical monocytes': 8,
    'Plasmacytoid Dendritic cells': 9,
    'Platelets': 10,
    'Pre-B cells': 11,
    'Progenitor cells': 12
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
how_many = 4880
indices = np.random.choice(len(adata), how_many, replace=False)
print(indices[:10])

unselected = np.setdiff1d(np.arange(len(adata)), indices)
print(unselected[:10])

# Create a TensorDataset from your AnnData object
X_tensor = torch.Tensor(adata.X[indices])#
cell_types_tensor = cell_types_tensor[indices]#


# Save X_tensor and cell_types_tensor as recon_counts_u.npy and labels_u.npy.
np.save('recon_counts_u.npy', X_tensor)
np.save('labels_u.npy', cell_types_tensor)

dataset = TensorDataset(X_tensor, cell_types_tensor)
mini_batch = how_many

dataloader = DataLoader(dataset, batch_size=mini_batch, shuffle=False)

input_dim = 36601
hidden_dim = 700
z_dim = 13

__all__ = ["mapping_dict", "dataloader", 'X_tensor', "cell_types_tensor", "mini_batch"]
