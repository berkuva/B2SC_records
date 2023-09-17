import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load data and labels
data = np.load('recon_counts.npy')
adata = sc.AnnData(X=data)
labels = np.load('labels.npy')

# Given label and color maps

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

# set label_map as the inverse of mapping_dict
label_map = {v: k for k, v in mapping_dict.items()}

color_map = {
    'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
    'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'purple', 'Naive CD8+ T cells': 'blue', 'Myeloid Dendritic cells': 'green',
    'Platelets': 'yellow'
}

# import pdb;pdb.set_trace()
for i in range(len(list(color_map.keys()))):
    print(str(list(color_map.keys())[i])+":"+str(np.unique(labels, return_counts=True)[1][i]/np.unique(labels, return_counts=True)[1].sum()))

# Naive B cells:0.2037037037037037
# Non-classical monocytes:0.1237037037037037
# Classical Monocytes:0.11259259259259259
# Natural killer  cells:0.10740740740740741
# CD8+ NKT-like cells:0.1474074074074074
# Memory CD4+ T cells:0.19592592592592592
# Naive CD8+ T cells:0.09074074074074075
# Myeloid Dendritic cells:0.012592592592592593
# Platelets:0.005925925925925926



# Assign labels to the adata object
# import pdb;pdb.set_trace()
adata.obs['labels'] = labels.astype(str)
adata.obs['labels'] = adata.obs['labels'].map(label_map).astype('category')
unique_labels = adata.obs['labels'].cat.categories

# Assign colors to labels
adata.uns['labels_colors'] = [color_map[label] for label in unique_labels]

# Normalize and scale data
sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.scale(adata, max_value=5000)

# Compute PCA and UMAP using Scanpy
sc.pp.pca(adata, n_comps=100)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=100) 
sc.tl.umap(adata)

# Plot UMAP using Scanpy
sc.settings.figsize = (12, 12)
sc.pl.umap(adata, color='labels', legend_loc=None, frameon=True, show=False)

# Add legend to the right of the plot
legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_map[label]) for label in unique_labels]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)

# Save the plot
plt.savefig("adjusted_umap_scanpy.png", bbox_inches='tight')

# Ensure the plot displays
plt.show()
