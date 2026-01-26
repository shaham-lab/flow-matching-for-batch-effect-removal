"""Configuration settings for SFMBER pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for SFMBER."""
    
    # Data settings
    batch_key: str = "batch"
    n_top_genes: int = 2000
    
    # scGen settings
    scgen_latent_dim: int = 100
    scgen_n_epochs: int = 100
    scgen_batch_size: int = 256
    scgen_learning_rate: float = 0.001
    
    # Flow matching settings
    flow_hidden_dims: list = None
    flow_n_epochs: int = 100
    flow_batch_size: int = 256
    flow_learning_rate: float = 0.001
    
    # OT settings
    ot_reg: float = 0.1  # Entropic regularization for OT
    
    # Inference settings
    n_steps: int = 100  # Number of steps for ODE integration
    
    # Device
    device: str = "cpu"
    
    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.flow_hidden_dims is None:
            self.flow_hidden_dims = [256, 256, 256]

