"""Loss functions for SFMBER training."""

import torch
import torch.nn as nn


def cfm_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor
) -> torch.Tensor:
    """
    Conditional Flow Matching loss.
    
    Args:
        v_pred: Predicted vector field [batch_size, latent_dim]
        v_target: Target vector field [batch_size, latent_dim]
        
    Returns:
        Scalar loss value
    """
    return torch.mean((v_pred - v_target) ** 2)

