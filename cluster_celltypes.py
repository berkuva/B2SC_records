import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


if __name__ == "__main__":
    from configure import configure
    
    data_dir = "/u/hc2kc/scVAE/pbmc1k/data/" # "path_to_data_directory"  # Please provide the correct path here
    barcode_path = data_dir+'barcode_to_celltype.csv' #"path_to_barcode_path"  # And also the correct path here

    args = configure(data_dir, barcode_path)

    # Load data and labels
    data = np.load('recon_counts.npy')
    adata = sc.AnnData(X=data)
    labels = np.load('labels.npy')


    # import pdb;pdb.set_trace()
    for i in range(len(list(args.color_map.keys()))):
        if list(args.label_map.keys())[i] in labels:
            print(str(list(args.color_map.keys())[i])+":"+str(np.unique(labels, return_counts=True)[1][i]/np.unique(labels, return_counts=True)[1].sum()))

    adata.obs['labels'] = labels.astype(str)
    adata.obs['labels'] = adata.obs['labels'].map(args.label_map).astype('category')
    unique_labels = adata.obs['labels'].cat.categories

    # Assign colors to labels
    adata.uns['labels_colors'] = [args.color_map[label] for label in unique_labels]

    # Normalize and scale data
    sc.pp.normalize_total(adata, target_sum=1e5)
    sc.pp.scale(adata, max_value=5000)

    # Compute PCA and UMAP using Scanpy
    sc.pp.pca(adata, n_comps=100)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=100) 
    sc.tl.tsne(adata)

    # Plot UMAP using Scanpy
    sc.settings.figsize = (12, 12)
    # sc.pl.umap(adata, color='labels', legend_loc=None, frameon=True, show=False)
    sc.pl.tsne(adata, color='labels', legend_loc=None, frameon=True, show=False)

    # Add legend to the right of the plot
    legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=args.color_map[label]) for label in unique_labels]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
    plt.title("TSNE Generated", fontsize=15)
    # Save the plot
    plt.savefig("pbmc_tsne.png", bbox_inches='tight', dpi=300)

    # Ensure the plot displays
    plt.show()


