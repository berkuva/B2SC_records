import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load data and labels
data = np.load('recon_counts.npy')
# data = data[:2700]
adata = sc.AnnData(X=data)
labels = np.load('labels.npy')
# labels = labels[:2700]
# np.save('recon_counts.npy', np.array(data))
# np.save('labels.npy', np.array(labels))



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

label_map = {v: k for k, v in mapping_dict.items()}

color_map = {
    'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
    'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Myeloid Dendritic cells': 'green',
    'Platelets': 'yellow'
}

for i in range(len(list(color_map.keys()))):
    print(str(list(color_map.keys())[i]) + ":" + str(np.unique(labels, return_counts=True)[1][i] / np.unique(labels, return_counts=True)[1].sum()))

# Assign labels to the adata object
adata.obs['labels'] = labels.astype(str)
adata.obs['labels'] = adata.obs['labels'].map(label_map).astype('category')
unique_labels = adata.obs['labels'].cat.categories

# Assign colors to labels
adata.uns['labels_colors'] = [color_map[label] for label in unique_labels]

# # Normalize and scale data
sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.scale(adata, max_value=5000)

# Compute PCA and t-SNE using Scanpy
sc.pp.pca(adata, n_comps=100)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=100) 
sc.tl.tsne(adata)  # Here's the t-SNE calculation 

# Plot t-SNE using Scanpy
sc.settings.figsize = (12, 12)
sc.pl.tsne(adata, color='labels', legend_loc=None, frameon=True, show=False)

# Add legend to the right of the plot
legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_map[label]) for label in unique_labels]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
plt.title("PBMC 3K Generated", fontsize=15)
# Save the plot
plt.savefig("pbmc3k.png", bbox_inches='tight', dpi=300)











# import scanpy as sc
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# # import torch
# # import paired_dataset
# # from paired_dataset import *

# # Load data and labels
# data = np.load('recon_counts.npy')
# adata = sc.AnnData(X=data)
# labels = np.load('labels.npy')

# # mean_variance_dict = {label.item(): (paired_dataset.X_tensor[paired_dataset.cell_types_tensor == label].mean(dim=0),\
# #                                      paired_dataset.X_tensor[paired_dataset.cell_types_tensor == label].var(dim=0, unbiased=True))\
# #                      for label in torch.unique(paired_dataset.cell_types_tensor)}


# # labels_tensor = torch.tensor(labels)
# # # Unique labels
# # unique_labels = torch.unique(labels_tensor)

# # # Process each label separately
# # for label in unique_labels:
# #     # Get the current label's data
# #     label_indices = (labels_tensor == label)
# #     label_data = data[label_indices]
    
# #     # Get the desired mean and variance for this label
# #     desired_mean, desired_variance = mean_variance_dict[label.item()]
    
# #     # Compute the current mean and variance
# #     current_mean = label_data.mean(axis=0)
# #     current_variance = label_data.var(axis=0, ddof=1)  # ddof=1 for unbiased variance
    
# #     # Convert numpy arrays to tensors for the following calculations
# #     current_mean = torch.tensor(current_mean)
# #     current_variance = torch.tensor(current_variance)
# #     desired_mean = torch.tensor(desired_mean)
# #     desired_variance = torch.tensor(desired_variance)
    
# #     # Compute scaling and shifting factors
# #     scale_factor = torch.sqrt(desired_variance / current_variance)
# #     shift_value = desired_mean - current_mean * scale_factor
    
# #     # Adjust the data for this label
# #     data[label_indices] = (label_data * scale_factor.numpy() + shift_value.numpy())


# # Now, 'data' should have the desired per-label mean and variance
# # import pdb;pdb.set_trace()
# # data = data.astype(np.int64)
# # data = data.clip(min=0)

# # # Convert labels to a tensor if it's a numpy array
# # labels_tensor = torch.tensor(labels) if isinstance(labels, np.ndarray) else labels

# # # Unique labels
# # unique_labels = torch.unique(labels_tensor)

# # # Check mean and variance for each label
# # for label in unique_labels:
# #     label_indices = (labels_tensor == label)
# #     label_data = data[label_indices]
    
# #     # Compute mean and variance
# #     computed_mean = label_data.mean(axis=0)  # Assuming label_data is a numpy array
# #     computed_variance = label_data.var(axis=0, ddof=1)  # Assuming label_data is a numpy array, ddof=1 for unbiased variance
    
# #     # Get the desired mean and variance
# #     desired_mean, desired_variance = mean_variance_dict[label.item()]
    
# #     # Convert desired mean and variance to numpy arrays if they are tensors
# #     desired_mean = desired_mean.numpy() if isinstance(desired_mean, torch.Tensor) else desired_mean
# #     desired_variance = desired_variance.numpy() if isinstance(desired_variance, torch.Tensor) else desired_variance
    
# #     # Print the results
# #     print(f'Label: {label.item()}')
# #     print(f'Computed mean: {computed_mean}, Desired mean: {desired_mean}')
# #     print(f'Computed variance: {computed_variance}, Desired variance: {desired_variance}')
# #     print(f'Mean match: {np.allclose(computed_mean, desired_mean)}, Variance match: {np.allclose(computed_variance, desired_variance)}')
# #     print('-' * 50)  # Print a separator line
# # # import pdb;pdb.set_trace()



# mapping_dict = {
#     'Naive B cells': '0',
#     'Non-classical monocytes': '1',
#     'Classical Monocytes': '2',
#     'Natural killer  cells': '3',
#     'CD8+ NKT-like cells': '4',
#     'Memory CD4+ T cells': '5',
#     'Naive CD8+ T cells': '6',
#     'Myeloid Dendritic cells': '7',
#     'Platelets': '8'
# }

# # set label_map as the inverse of mapping_dict
# label_map = {v: k for k, v in mapping_dict.items()}

# color_map = {
#     'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
#     'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Myeloid Dendritic cells': 'green',
#     'Platelets': 'yellow'
# }

# # import pdb;pdb.set_trace()
# for i in range(len(list(color_map.keys()))):
#     print(str(list(color_map.keys())[i])+":"+str(np.unique(labels, return_counts=True)[1][i]/np.unique(labels, return_counts=True)[1].sum()))

# # Naive B cells:0.2037037037037037
# # Non-classical monocytes:0.1237037037037037
# # Classical Monocytes:0.11259259259259259
# # Natural killer  cells:0.10740740740740741
# # CD8+ NKT-like cells:0.1474074074074074
# # Memory CD4+ T cells:0.19592592592592592
# # Naive CD8+ T cells:0.09074074074074075
# # Myeloid Dendritic cells:0.012592592592592593
# # Platelets:0.005925925925925926



# # Assign labels to the adata object
# # import pdb;pdb.set_trace()
# adata.obs['labels'] = labels.astype(str)
# adata.obs['labels'] = adata.obs['labels'].map(label_map).astype('category')
# unique_labels = adata.obs['labels'].cat.categories

# # Assign colors to labels
# adata.uns['labels_colors'] = [color_map[label] for label in unique_labels]

# # Normalize and scale data
# sc.pp.normalize_total(adata, target_sum=1e5)
# sc.pp.scale(adata, max_value=5000)

# # Compute PCA and UMAP using Scanpy
# sc.pp.pca(adata, n_comps=100)
# sc.pp.neighbors(adata, n_neighbors=10, n_pcs=100) 
# sc.tl.umap(adata)

# # Plot UMAP using Scanpy
# sc.settings.figsize = (12, 12)
# sc.pl.umap(adata, color='labels', legend_loc=None, frameon=True, show=False)

# # Add legend to the right of the plot
# legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_map[label]) for label in unique_labels]
# plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)

# # Save the plot
# plt.savefig("pbmc3k_generated.png", bbox_inches='tight')

# # Ensure the plot displays
# plt.show()
