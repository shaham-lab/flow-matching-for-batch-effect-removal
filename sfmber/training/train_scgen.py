"""Training script for scGen VAE."""

from typing import Optional
from anndata import AnnData
from sfmber.models.scgen_wrapper import ScGenWrapper
from sfmber.utils.config import Config


def train_scgen(
    adata: AnnData,
    config: Config,
    model: Optional[ScGenWrapper] = None
) -> ScGenWrapper:
    """
    Train scGen VAE on combined batches.
    
    Args:
        adata: Combined AnnData from all batches
        config: Configuration object
        model: Optional pre-initialized model
        
    Returns:
        Trained ScGenWrapper model
    """
    if model is None:
        n_genes = adata.n_vars
        model = ScGenWrapper(
            n_genes=n_genes,
            latent_dim=config.scgen_latent_dim,
            device=config.device
        )
    
    # Prepare data for scGen
    from sfmber.data.preprocess import prepare_for_scgen
    adata_prep = prepare_for_scgen(adata)
    
    # Train model
    model.train(
        adata=adata_prep,
        n_epochs=config.scgen_n_epochs,
        batch_size=config.scgen_batch_size,
        learning_rate=config.scgen_learning_rate
    )
    
    return model

