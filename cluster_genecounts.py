import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load data and labels
data = np.load('recon_counts.npy')
adata = sc.AnnData(X=data)
labels = np.load('labels.npy')
# import pdb;pdb.set_trace()

# Given label and color maps
mapping_dict = {
                'Basophils': '0',
                'CD8+ NKT-like cells': '1',
                'Classical Monocytes': '2',
                'Erythroid-like and erythroid precursor cells': '3',
                'Memory CD4+ T cells': '4',
                'Naive B cells': '5',
                'Naive CD4+ T cells': '6',
                'Natural killer  cells': '7',
                'Non-classical monocytes': '8',
                'Plasmacytoid Dendritic cells': '9',
                'Platelets': '10',
                'Pre-B cells': '11',
                'Progenitor cells': '12'
                }

# set label_map as the inverse of mapping_dict
label_map = {v: k for k, v in mapping_dict.items()}


color_map = {
    'Basophils': 'darkred',
    'CD8+ NKT-like cells': 'pink',
    'Classical Monocytes': 'orange',
    'Erythroid-like and erythroid precursor cells': 'maroon',
    'Memory CD4+ T cells': 'magenta',
    'Naive B cells': 'red',
    'Naive CD4+ T cells': 'slateblue',
    'Natural killer  cells': 'cyan',
    'Non-classical monocytes': 'black',
    'Plasmacytoid Dendritic cells': 'lime',
    'Platelets': 'yellow',
    'Pre-B cells': 'cornflowerblue',
    'Progenitor cells': 'darkgreen'
}



# import pdb;pdb.set_trace()
for i in range(len(list(color_map.keys()))):
    print(str(list(color_map.keys())[i])+":"+str(np.unique(labels, return_counts=True)[1][i]/np.unique(labels, return_counts=True)[1].sum()))



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

# Compute PCA and t-SNE using Scanpy
sc.pp.pca(adata, n_comps=100)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=100) 
sc.tl.tsne(adata)

# Plot t-SNE using Scanpy
sc.settings.figsize = (12, 12)
sc.pl.tsne(adata, color='labels', legend_loc=None, frameon=True, show=False)

# Add legend to the right of the plot
legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_map[label]) for label in unique_labels]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
plt.title("PBMC 5K Generated", fontsize=15)
# Save the plot
plt.savefig("pbmc5k.png", bbox_inches='tight', dpi=300)
