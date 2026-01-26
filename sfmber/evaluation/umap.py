"""UMAP visualization utilities."""

import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData
from typing import Optional, Tuple


def compute_umap(adata: AnnData, n_neighbors: int = 15, min_dist: float = 0.5) -> AnnData:
    """
    Compute UMAP embedding for AnnData.
    
    Args:
        adata: AnnData object
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        
    Returns:
        AnnData with UMAP coordinates in adata.obsm['X_umap']
    """
    adata = adata.copy()
    
    # Compute PCA if not present
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    
    # Compute UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=50)
    sc.tl.umap(adata, min_dist=min_dist)
    
    return adata


def plot_umap_comparison(
    adata_before: AnnData,
    adata_after: AnnData,
    batch_key: str = "batch",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot UMAP before and after batch correction.
    
    Args:
        adata_before: Original data
        adata_after: Corrected data
        batch_key: Key for batch labels
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Compute UMAP for both
    adata_before = compute_umap(adata_before)
    adata_after = compute_umap(adata_after)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot before
    sc.pl.umap(
        adata_before,
        color=batch_key,
        ax=axes[0],
        show=False,
        title="Before Correction"
    )
    
    # Plot after
    sc.pl.umap(
        adata_after,
        color=batch_key,
        ax=axes[1],
        show=False,
        title="After Correction"
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

