"""Optimal Transport pairing for flow matching."""

import torch
import numpy as np
from typing import Tuple
from ot import sinkhorn


def compute_ot_pairing(
    z0: torch.Tensor,
    z1: torch.Tensor,
    reg: float = 0.1,
    max_iter: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal transport pairing between two sets of latent points.
    
    Uses Sinkhorn algorithm to find empirical coupling π* that pairs
    points from Z0 to Z1 with minimal transport cost.
    
    Args:
        z0: Source latent points [N0, latent_dim]
        z1: Target latent points [N1, latent_dim]
        reg: Entropic regularization parameter
        max_iter: Maximum iterations for Sinkhorn
        
    Returns:
        Tuple of (paired_z0, paired_z1) with same number of points
    """
    # Convert to numpy for POT library
    z0_np = z0.detach().cpu().numpy()
    z1_np = z1.detach().cpu().numpy()
    
    # Compute pairwise squared Euclidean distances
    n0, d = z0_np.shape
    n1, _ = z1_np.shape
    
    # Cost matrix: C[i,j] = ||z0[i] - z1[j]||^2
    z0_expanded = z0_np[:, np.newaxis, :]  # [N0, 1, d]
    z1_expanded = z1_np[np.newaxis, :, :]  # [1, N1, d]
    cost_matrix = np.sum((z0_expanded - z1_expanded) ** 2, axis=2)  # [N0, N1]
    
    # Uniform marginals
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    
    # Solve OT problem with Sinkhorn
    pi = sinkhorn(a, b, cost_matrix, reg, numItermax=max_iter)
    
    # Extract paired indices using greedy matching from coupling
    # For each point in z0, find best match in z1
    paired_indices_0 = []
    paired_indices_1 = []
    
    # Use Hungarian-like matching from coupling matrix
    remaining_0 = set(range(n0))
    remaining_1 = set(range(n1))
    
    # Sort by coupling strength
    pairs = []
    for i in range(n0):
        for j in range(n1):
            if pi[i, j] > 1e-6:  # Threshold for meaningful coupling
                pairs.append((pi[i, j], i, j))
    
    pairs.sort(reverse=True)  # Sort by coupling strength
    
    # Greedy matching
    for _, i, j in pairs:
        if i in remaining_0 and j in remaining_1:
            paired_indices_0.append(i)
            paired_indices_1.append(j)
            remaining_0.remove(i)
            remaining_1.remove(j)
    
    # If sizes don't match, pad with nearest neighbors
    min_size = min(len(paired_indices_0), len(paired_indices_1))
    if len(paired_indices_0) < n0 or len(paired_indices_1) < n1:
        # For remaining points, find nearest neighbors
        for i in remaining_0:
            j = np.argmin(cost_matrix[i, :])
            paired_indices_0.append(i)
            paired_indices_1.append(j)
        
        for j in remaining_1:
            i = np.argmin(cost_matrix[:, j])
            paired_indices_0.append(i)
            paired_indices_1.append(j)
    
    # Ensure equal sizes
    min_len = min(len(paired_indices_0), len(paired_indices_1))
    paired_indices_0 = paired_indices_0[:min_len]
    paired_indices_1 = paired_indices_1[:min_len]
    
    # Extract paired points
    paired_z0 = z0[paired_indices_0]
    paired_z1 = z1[paired_indices_1]
    
    return paired_z0, paired_z1

