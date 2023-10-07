import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

data_dir = "/u/hc2kc/scVAE/pbmc1k/data/"
adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

barcode_path = data_dir+'barcode_to_celltype.csv'
barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
barcodes_with_labels.columns = ['barcodes', 'labels']

barcodes_with_labels = barcodes_with_labels[
    (barcodes_with_labels['labels'] != 'Unknown')
]

cleaned_labels = barcodes_with_labels['labels'].values
labels = cleaned_labels

filtered_barcodes = barcodes_with_labels['barcodes'].values
adata = adata[adata.obs.index.isin(filtered_barcodes)]

# Add the labels to adata.obs
adata.obs['labels'] = barcodes_with_labels.set_index('barcodes').loc[adata.obs.index, 'labels']

mapping_dict = {
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

adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict).astype('category')

label_map = {v: k for k, v in mapping_dict.items()}
adata.obs['labels'] = adata.obs['labels'].map(label_map).astype('category')

color_map = {
    'Naive B cells': 'red',
    'Non-classical monocytes': 'black',
    'Classical Monocytes': 'orange',
    'Natural killer  cells': 'cyan',
    'CD8+ NKT-like cells': 'pink',
    'Memory CD4+ T cells': 'magenta',
    'Naive CD8+ T cells': 'blue',
    'Platelets': 'yellow',
    'Plasmacytoid Dendritic cells':'lime',
    'Pre-B cells':'cornflowerblue'
}

unique_labels = adata.obs['labels'].cat.categories
adata.uns['labels_colors'] = [color_map[label] for label in unique_labels]

sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.scale(adata, max_value=5000)

sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.tsne(adata)

sc.settings.figsize = (12, 12)
sc.pl.tsne(adata, color='labels', legend_loc=None, frameon=True, show=False)

legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_map[label]) for label in unique_labels]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
plt.title('PBMC 1K Raw', fontsize=15)
plt.savefig("pbmc1k_tsne.png", bbox_inches='tight', dpi=300)
