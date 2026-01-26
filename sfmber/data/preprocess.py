"""Data preprocessing utilities for SFMBER."""

import scanpy as sc
import numpy as np
from anndata import AnnData
from typing import Optional


def preprocess_data(
    adata: AnnData,
    n_top_genes: int = 2000,
    normalize: bool = True,
    log_transform: bool = True,
    scale: bool = False
) -> AnnData:
    """
    Preprocess single-cell data.
    
    Args:
        adata: AnnData object to preprocess
        n_top_genes: Number of highly variable genes to select
        normalize: Whether to normalize to 10k counts per cell
        log_transform: Whether to apply log1p transformation
        scale: Whether to scale to unit variance
        
    Returns:
        Preprocessed AnnData object
    """
    adata = adata.copy()
    
    # Calculate QC metrics if not present
    if "n_genes_by_counts" not in adata.var.columns:
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Normalize to 10k counts per cell
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    if log_transform:
        sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=False)
    
    # Filter to HVGs
    adata = adata[:, adata.var.highly_variable].copy()
    
    # Scale to unit variance (optional)
    if scale:
        sc.pp.scale(adata, max_value=10)
    
    return adata


def prepare_for_scgen(adata: AnnData) -> AnnData:
    """
    Prepare data specifically for scGen training.
    
    Args:
        adata: AnnData object
        
    Returns:
        Prepared AnnData object
    """
    adata = adata.copy()
    
    # Ensure raw counts are stored
    if adata.raw is None:
        adata.raw = adata
    
    # Preprocess for scGen (normalize and log transform)
    adata = preprocess_data(adata, normalize=True, log_transform=True, scale=False)
    
    return adata

