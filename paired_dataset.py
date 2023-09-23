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

data_dir = "/u/hc2kc/scVAE/paired/pbmc10k/"

# adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
adata = sc.read_h5ad("/u/hc2kc/scVAE/paired/pbmc10k/pbmc10k.h5ad")
# import pdb;pdb.set_trace()
barcode_path = '/u/hc2kc/scVAE/paired/pbmc10k/barcode_to_celltype.csv'
barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
barcodes_with_labels.columns = ['barcodes', 'labels']
labels = barcodes_with_labels['labels'].values
assert len(labels) == len(adata)

# import pdb;pdb.set_trace()
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
gene_std_devs = np.std(adata.X, axis=0)

# Select top variable genes based on standard deviation
top_variable_genes = np.argsort(gene_std_devs)

# Subset your data to include only these genes
adata = adata[:, top_variable_genes]

keys = [
    'CD8+ NKT-like cells', 'Classical Monocytes',
    'Effector CD4+ T cells', 'Macrophages', 'Myeloid Dendritic cells',
    'Naive B cells', 'Naive CD4+ T cells', 'Naive CD8+ T cells',
    'Natural killer  cells', 'Non-classical monocytes',
    'Plasma B cells', 'Plasmacytoid Dendritic cells', 'Pre-B cells'
]

# Create a dictionary using dictionary comprehension
mapping_dict = {key: value for value, key in enumerate(keys)}

# Now key_value_dict contains your keys with values from 0 to 12
print(mapping_dict)
# {'CD8+ NKT-like cells': 0,\
# 'Classical Monocytes': 1,\
# 'Effector CD4+ T cells': 2,\
# 'Macrophages': 3,\
# 'Myeloid Dendritic cells': 4,\
# 'Naive B cells': 5,\
# 'Naive CD4+ T cells': 6,\
# 'Naive CD8+ T cells': 7,\
# 'Natural killer  cells': 8,\
# 'Non-classical monocytes': 9,\
# 'Plasma B cells': 10,\
# 'Plasmacytoid Dendritic cells': 11,\
# 'Pre-B cells': 12}

# Apply mapping to the 'labels' column of adata.obs
# import pdb;pdb.set_trace()
adata.obs['labels'] = adata.obs['labels'].astype('category')
adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)

# Filter out rows with the label "Unknown" (which has been mapped to '15')
# adata = adata[adata.obs['labels'] != '15', :]

# Continue with your existing code
# labels = adata.obs['labels'].cat.codes.values
# import pdb;pdb.set_trace()
cell_types_tensor = torch.LongTensor(np.copy(adata.obs['labels']))

# Randomly select  cells from your AnnData object
how_many = 6156#12313
indices = np.random.choice(len(adata), how_many, replace=False)
print(indices[:10])

# unselected = np.setdiff1d(np.arange(len(adata)), indices)
# print(unselected[:10])

# Create a TensorDataset from your AnnData object
X_tensor = torch.Tensor(adata.X[indices])#
cell_types_tensor = cell_types_tensor[indices]#
dataset = TensorDataset(X_tensor, cell_types_tensor)
mini_batch = 6156 

dataloader = DataLoader(dataset, batch_size=mini_batch, shuffle=False)
# import pdb;pdb.set_trace()
input_dim = 36601
hidden_dim = 700
z_dim = 15

__all__ = ["mapping_dict", "dataloader", 'X_tensor', "cell_types_tensor", "mini_batch"]


# color_map = {
#     '0': 'pink',
#     '1': 'orange',
#     '2': 'grey',
#     '3': 'tan',
#     '4': 'purple',
#     '5': 'green',
#     '6': 'red',
#     '7': 'slateblue',
#     '8': 'blue',
#     '9': 'cyan',
#     '10': 'black',
#     '11': 'lime',
#     '12': 'yellow',
#     '13': 'cornflowerblue',
#     '14': 'lavender',
#     '15': 'brown'
# }

# # Create a new AnnData object for unselected cells
# adata_unselected = adata[unselected].copy()

# # Update labels for unselected cells
# labels_unselected = labels[unselected]
# adata_unselected.obs['labels'] = pd.Categorical(labels_unselected)



# # Create a new AnnData object for the selected cells
# adata_selected = adata[indices].copy()

# # Update labels for selected cells
# labels_selected = labels[indices]
# adata_selected.obs['labels'] = pd.Categorical(labels_selected)

# # Concatenate the two AnnData objects
# adata_final = adata_selected.concatenate(adata_unselected, index_unique=None, join='outer')

# sc.pp.neighbors(adata_final)
# # Finally, compute the UMAP
# sc.tl.umap(adata_final)

# # Reverse mapping_dict to get a dict from integer strings to cell types
# reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}

# # Create legend handles
# legend_handles = [mpatches.Patch(color=color_map[str(i)], label=reverse_mapping_dict[str(i)]) for i in range(len(mapping_dict))]

# # Generate colors list based on labels
# colors = [color_map[label] for label in adata_final.obs['labels'].astype(str).tolist()]

# # Plot UMAP
# fig, ax = plt.subplots()
# scatter = ax.scatter(adata_final.obsm['X_umap'][:,0], adata_final.obsm['X_umap'][:,1], c=colors, alpha=0.6)

# # Create legend
# legend = ax.legend(handles=legend_handles, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')

# # Adjust the plot dimensions to make room for the legend
# plt.subplots_adjust(right=0.55)

# # Get current time
# current_time = datetime.now()

# # Format current time
# formatted_time = current_time.strftime('%B%d%H%M%S')

# # Save the plot
# fig.savefig(f'UMAP_plot_{formatted_time}.png', bbox_extra_artists=(legend,), bbox_inches='tight')

