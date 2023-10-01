import scanpy as sc
import pandas as pd
import numpy as np
# grand_path = '/u/hc2kc/scVAE/pbmc3k/'
grand_path = '/u/hc2kc/scVAE/'


# Load features and data as before
features_3k = pd.read_csv('/u/hc2kc/hg19/genes.tsv', sep='\t', header=None)
# features_3k = pd.read_csv('/u/hc2kc/hg19/features.tsv', sep='\t', header=None)
features_3k.drop([1], axis=1, inplace=True)
# features_3k.drop([1,2,3,4,5], axis=1, inplace=True)

features_3k.columns = ["gene_name"]

# Assuming grand_path is defined
unused_3k = np.load(grand_path + 'recon_counts.npy')
unused_labels = np.load(grand_path + 'labels.npy')

# Create an AnnData object
# import pdb; pdb.set_trace()
var_dataframe = pd.DataFrame(index=features_3k['gene_name'].head(unused_3k.shape[1]))
adata = sc.AnnData(X=unused_3k, obs=pd.DataFrame(index=unused_labels), var=var_dataframe)
# adata = sc.AnnData(X=unused_10k, obs=pd.DataFrame(index=unused_labels), var=pd.DataFrame(index=features_10k['gene_name']).iloc[len(unused_10k)])

# Optionally, if you have cell type labels, you can add them to the AnnData object
adata.obs['cell_type'] = unused_labels

adata.X = np.clip(adata.X, 0, None)
adata.X = np.round(adata.X).astype(int)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# Dimensionality reduction
sc.tl.pca(adata, svd_solver='arpack')

# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Convert the 'cell_type' column to 'category' dtype
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
# adata.obs['cell_type'] = unused_labels.astype(str)

# If you have cell type labels
sc.tl.rank_genes_groups(adata, groupby='cell_type', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)


# If you want to save the results of the marker gene analysis to a CSV file:
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
marker_genes_df = pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']})

marker_genes_df.to_csv('3k_marker_genes.csv')