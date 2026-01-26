"""Data loading utilities for SFMBER."""

import scanpy as sc
from anndata import AnnData
from typing import Tuple, Optional


def load_adata(file_path: str) -> AnnData:
    """
    Load AnnData object from file.
    
    Args:
        file_path: Path to the data file (h5ad, csv, etc.)
        
    Returns:
        AnnData object containing the dataset
    """
    adata = sc.read(file_path)
    return adata


def split_batches(
    adata: AnnData,
    batch_key: str = "batch",
    source_batch: Optional[str] = None,
    target_batch: Optional[str] = None
) -> Tuple[AnnData, AnnData]:
    """
    Split dataset into source batch B0 and target batch B1.
    
    Args:
        adata: Combined AnnData object
        batch_key: Key in adata.obs containing batch labels
        source_batch: Label for source batch. If None, uses first batch.
        target_batch: Label for target batch. If None, uses second batch.
        
    Returns:
        Tuple of (source_adata, target_adata)
    """
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    
    unique_batches = adata.obs[batch_key].unique()
    
    if source_batch is None:
        source_batch = unique_batches[0]
    if target_batch is None:
        if len(unique_batches) < 2:
            raise ValueError("Need at least 2 batches for correction")
        target_batch = unique_batches[1]
    
    source_adata = adata[adata.obs[batch_key] == source_batch].copy()
    target_adata = adata[adata.obs[batch_key] == target_batch].copy()
    
    return source_adata, target_adata

