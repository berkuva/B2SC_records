import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

# Load data.
data = np.load('recon_counts.npy')
adata = sc.AnnData(X=data)
labels = np.load('labels.npy')

label_map = {
    '0': 'Naive B cells',
    '1': 'Non-classical monocytes',
    '2': 'Classical Monocytes',
    '3': 'Natural killer cells',
    '4': 'Naive CD8+ T cells',
    '5': 'Memory CD4+ T cells',
    '6': 'CD8+ NKT-like cells',
    '7': 'Myeloid Dendritic cells',
    '8': 'Platelets',
}

# Convert integer labels to strings, map them using label_map, and assign back to adata
adata.obs['labels'] = pd.Series(labels.astype(str)).map(label_map).values

# Normalize and scale data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.scale(adata, max_value=100)

# Run PCA
sc.tl.pca(adata, svd_solver='arpack')

# Compute the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Compute t-SNE embeddings
sc.tl.tsne(adata)

color_map = {
    'Naive B cells': 'red',
    'Classical Monocytes': 'orange',
    'Platelets': 'yellow',
    'Myeloid Dendritic cells': 'green',
    'Naive CD8+ T cells': 'blue',
    'Non-classical monocytes': 'black',
    'Memory CD4+ T cells': 'purple',
    'CD8+ NKT-like cells': 'pink',
    'Natural killer cells': 'cyan'
}

# Plot t-SNE using the 'labels' column for coloring and the color_map as the palette
sc.pl.tsne(adata, color='labels', save='tsne.png', palette=color_map)
