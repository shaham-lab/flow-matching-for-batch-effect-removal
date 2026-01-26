"""Wrapper for scGen VAE encoder/decoder."""

import torch
import numpy as np
from typing import Optional
from anndata import AnnData
import scgen


class ScGenWrapper:
    """
    Wrapper for scGen VAE that exposes only encode/decode functionality.
    """
    
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize scGen wrapper.
        
        Args:
            n_genes: Number of input genes
            latent_dim: Latent space dimensionality
            device: Device to run on ('cpu' or 'cuda')
        """
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.device = device
        self.model: Optional[scgen.SCGEN] = None
        self.is_trained = False
    
    def build_model(self, adata: AnnData) -> None:
        """
        Build scGen model from AnnData.
        
        Args:
            adata: AnnData object used to initialize scGen
        """
        self.model = scgen.SCGEN(adata, n_genes=self.n_genes)
        self.is_trained = False
    
    def train(
        self,
        adata: AnnData,
        n_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001
    ) -> None:
        """
        Train scGen VAE.
        
        Args:
            adata: Training data
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        if self.model is None:
            self.build_model(adata)
        
        self.model.train(
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stopping=True,
            early_stopping_patience=25,
            use_gpu=(self.device == "cuda")
        )
        self.is_trained = True
    
    def encode(self, adata: AnnData) -> torch.Tensor:
        """
        Encode data to latent space.
        
        Args:
            adata: AnnData object to encode
            
        Returns:
            Latent representations as torch.Tensor
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before encoding")
        
        # Get latent representation
        latent = self.model.to_latent(adata.X)
        
        # Convert to torch tensor
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).float()
        else:
            latent = torch.tensor(latent, dtype=torch.float32)
        
        return latent.to(self.device)
    
    def decode(self, z: torch.Tensor) -> np.ndarray:
        """
        Decode latent representation to expression space.
        
        Args:
            z: Latent representations (torch.Tensor)
            
        Returns:
            Decoded expression as numpy array
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before decoding")
        
        # Convert to numpy if needed
        if isinstance(z, torch.Tensor):
            z_np = z.detach().cpu().numpy()
        else:
            z_np = np.array(z)
        
        # Access scGen's VAE model decoder directly
        with torch.no_grad():
            z_tensor = torch.from_numpy(z_np).float()
            if self.device == "cuda":
                z_tensor = z_tensor.cuda()
                self.model.model.to(self.device)
            
            self.model.model.eval()
            # Access the decoder through model.model (scGen's internal structure)
            decoded = self.model.model.decoder(z_tensor)
            decoded_np = decoded.cpu().numpy()
        
        return decoded_np

