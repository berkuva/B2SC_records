import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

data_dir = "/u/hc2kc/hg19/"

adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

barcode_path = '/u/hc2kc/hg19/barcodes_with_labels.txt'
barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None)
barcodes_with_labels.columns = ['barcodes', 'labels']

adata.obs['barcodes'] = adata.obs.index
adata.obs = adata.obs.reset_index(drop=True)
adata.obs = adata.obs.merge(barcodes_with_labels, on='barcodes', how='left')

adata.X = adata.X.toarray()
adata.obs.index = adata.obs.index.astype(str)

# sc.pp.filter_cells(adata, min_genes=10)

# gene_std_devs = np.std(adata.X, axis=0)
# top_variable_genes = np.argsort(gene_std_devs)[-500:]

# adata = adata[:, top_variable_genes]

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

adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict).astype('category')
label_map = {v: k for k, v in mapping_dict.items()}

color_map = {
    'Naive B cells': 'red',
    'Non-classical monocytes': 'black',
    'Classical Monocytes': 'orange',
    'Natural killer  cells': 'cyan',
    'CD8+ NKT-like cells': 'pink',
    'Memory CD4+ T cells': 'magenta',
    'Naive CD8+ T cells': 'blue',
    'Myeloid Dendritic cells': 'green',
    'Platelets': 'yellow'
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
# Remove ticks.
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
# x label as TSNE1 and y label as TSNE2.
plt.xlabel('TSNE1', fontsize=13)
plt.ylabel('TSNE2', fontsize=13)

legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_map[label]) for label in unique_labels]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
# Set title to PBMC 3K Raw.
plt.title('PBMC 3K Raw', fontsize=15)
plt.savefig("pbmc_tsne.png", bbox_inches='tight', dpi=300)
