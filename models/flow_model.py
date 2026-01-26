"""Flow matching model for batch correction."""

import torch
import torch.nn as nn
from typing import Tuple


class VectorFieldMLP(nn.Module):
    """
    MLP that parameterizes the vector field v_theta(z, t).
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [256, 256, 256],
        activation: str = "swish"
    ):
        """
        Initialize vector field MLP.
        
        Args:
            latent_dim: Dimensionality of latent space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('swish', 'relu', 'tanh')
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Activation function
        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network: input is [z, t] -> output is vector field
        dims = [latent_dim + 1] + hidden_dims + [latent_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(self.activation)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute vector field v_theta(z, t).
        
        Args:
            z: Latent points [batch_size, latent_dim]
            t: Time values [batch_size, 1] or [batch_size]
            
        Returns:
            Vector field values [batch_size, latent_dim]
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Concatenate z and t
        zt = torch.cat([z, t], dim=1)
        
        # Forward through network
        v = self.net(zt)
        
        return v


class FlowMatchingModel:
    """
    Conditional Flow Matching model for batch correction.
    Uses linear interpolation ψ_t(z0, z1) = (1-t)z0 + t*z1.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [256, 256, 256],
        device: str = "cpu"
    ):
        """
        Initialize flow matching model.
        
        Args:
            latent_dim: Dimensionality of latent space
            hidden_dims: Hidden dimensions for vector field MLP
            device: Device to run on
        """
        self.latent_dim = latent_dim
        self.device = device
        
        self.vector_field = VectorFieldMLP(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(device)
    
    def linear_interpolation(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute linear interpolation ψ_t(z0, z1) = (1-t)z0 + t*z1.
        
        Args:
            z0: Source points [batch_size, latent_dim]
            z1: Target points [batch_size, latent_dim]
            t: Time values [batch_size, 1] or [batch_size]
            
        Returns:
            Interpolated points [batch_size, latent_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        return (1 - t) * z0 + t * z1
    
    def compute_cfm_loss(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Conditional Flow Matching loss.
        
        L_CFM = E_{t ~ U(0,1)} ||v_theta(ψ_t(z0, z1), t) - (z1 - z0)||^2
        
        Args:
            z0: Source points [batch_size, latent_dim]
            z1: Target points [batch_size, latent_dim]
            
        Returns:
            Scalar loss value
        """
        batch_size = z0.shape[0]
        
        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=self.device)
        
        # Compute interpolation
        zt = self.linear_interpolation(z0, z1, t)
        
        # Compute vector field
        v_pred = self.vector_field(zt, t.squeeze())
        
        # Target vector field is z1 - z0
        v_target = z1 - z0
        
        # Compute MSE loss
        loss = torch.mean((v_pred - v_target) ** 2)
        
        return loss
    
    def get_vector_field(self) -> nn.Module:
        """Get the vector field network."""
        return self.vector_field

