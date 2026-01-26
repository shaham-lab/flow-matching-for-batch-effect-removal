"""Batch correction inference using trained models."""

import torch
import numpy as np
from anndata import AnnData
from sfmber.models.scgen_wrapper import ScGenWrapper
from sfmber.models.flow_model import FlowMatchingModel
from sfmber.inference.integrate_flow import integrate_flow
from sfmber.utils.config import Config


def correct_batch(
    source_adata: AnnData,
    scgen_model: ScGenWrapper,
    flow_model: FlowMatchingModel,
    config: Config,
    n_steps: int = None
) -> AnnData:
    """
    Correct source batch to match target batch distribution.
    
    Process:
    1. Encode source batch to latent space: z0 = E(x0)
    2. Integrate vector field: z_corr = z0 + ∫[0 to 1] v_theta(z_t, t) dt
    3. Decode to expression space: x_corr = D(z_corr)
    
    Args:
        source_adata: Source batch AnnData to correct
        scgen_model: Trained scGen model
        flow_model: Trained flow matching model
        config: Configuration object
        n_steps: Number of integration steps (overrides config if provided)
        
    Returns:
        Corrected AnnData object
    """
    if n_steps is None:
        n_steps = config.n_steps
    
    # Step 1: Encode to latent space
    z0 = scgen_model.encode(source_adata)
    
    # Step 2: Integrate vector field
    def vector_field_fn(z, t):
        """Wrapper for vector field evaluation."""
        return flow_model.vector_field(z, t)
    
    z_corr = integrate_flow(
        z0,
        vector_field_fn,
        n_steps=n_steps,
        method="euler"
    )
    
    # Step 3: Decode to expression space
    x_corr = scgen_model.decode(z_corr)
    
    # Create corrected AnnData
    corrected_adata = source_adata.copy()
    corrected_adata.X = x_corr
    
    return corrected_adata

