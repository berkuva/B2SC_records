import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

data_dir = "/u/hc2kc/scVAE/pbmc5k/data/"

adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

barcode_path = '/u/hc2kc/scVAE/pbmc5k/data/barcode_to_celltype.tsv'
barcodes_with_labels = pd.read_csv(barcode_path, sep='\t', header=None)
barcodes_with_labels.columns = ['barcodes', 'labels']


adata.obs['barcodes'] = adata.obs.index
adata.obs = adata.obs.reset_index(drop=True)
adata.obs = adata.obs.merge(barcodes_with_labels, on='barcodes', how='left')

# Convert sparse matrix to dense matrix if necessary
adata.X = adata.X.toarray()
adata.obs.index = adata.obs.index.astype(str)


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

# import pdb;pdb.set_trace()
# Filter out rows with the label "Unknown" (which has been mapped to '15')
# adata = adata[adata.obs['labels'] != '13', :]
adata = adata[~adata.obs['labels'].isna(), :]

adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict).astype('category')

# set label_map as the inverse of mapping_dict
label_map = {v: k for k, v in mapping_dict.items()}



color_map = {
    'Basophils': 'goldenrod',
    'CD8+ NKT-like cells': 'pink',
    'Classical Monocytes': 'orange',
    'Erythroid-like and erythroid precursor cells': 'silver',
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


adata.obs['labels'] = adata.obs['labels'].map(label_map).astype('category')
unique_labels = adata.obs['labels'].cat.categories
adata.uns['labels_colors'] = [color_map[label] for label in unique_labels]

sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.scale(adata, max_value=5000)

sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.tsne(adata)  # Compute t-SNE instead of UMAP

sc.settings.figsize = (12, 12)
sc.pl.tsne(adata, color='labels', legend_loc=None, frameon=True, show=False)  # Plot t-SNE results

legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_map[label]) for label in unique_labels]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
# Set title to PBMC 3K Raw.
plt.title('PBMC 5K Raw', fontsize=15)
plt.savefig("pbmc5k_tsne.png", bbox_inches='tight', dpi=300)
