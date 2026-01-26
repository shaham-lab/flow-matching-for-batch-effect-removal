"""Training script for flow matching model."""

import torch
import torch.optim as optim
from typing import Tuple
from anndata import AnnData
from sfmber.models.scgen_wrapper import ScGenWrapper
from sfmber.models.flow_model import FlowMatchingModel
from sfmber.models.ot_pairing import compute_ot_pairing
from sfmber.utils.config import Config


def train_flow_model(
    source_adata: AnnData,
    target_adata: AnnData,
    scgen_model: ScGenWrapper,
    config: Config,
    flow_model: FlowMatchingModel = None
) -> FlowMatchingModel:
    """
    Train flow matching model for batch correction.
    
    Steps:
    1. Encode B0 and B1 to latent space
    2. Compute OT pairing
    3. Train flow model using CFM loss
    
    Args:
        source_adata: Source batch AnnData
        target_adata: Target batch AnnData
        scgen_model: Trained scGen model
        config: Configuration object
        flow_model: Optional pre-initialized flow model
        
    Returns:
        Trained FlowMatchingModel
    """
    # Encode to latent space
    print("Encoding batches to latent space...")
    z0 = scgen_model.encode(source_adata)
    z1 = scgen_model.encode(target_adata)
    
    # Compute OT pairing
    print("Computing optimal transport pairing...")
    paired_z0, paired_z1 = compute_ot_pairing(
        z0, z1, reg=config.ot_reg
    )
    
    # Initialize flow model if needed
    if flow_model is None:
        flow_model = FlowMatchingModel(
            latent_dim=config.scgen_latent_dim,
            hidden_dims=config.flow_hidden_dims,
            device=config.device
        )
    
    # Setup optimizer
    optimizer = optim.Adam(
        flow_model.vector_field.parameters(),
        lr=config.flow_learning_rate
    )
    
    # Training loop
    print("Training flow matching model...")
    n_pairs = len(paired_z0)
    n_batches = (n_pairs + config.flow_batch_size - 1) // config.flow_batch_size
    
    flow_model.vector_field.train()
    
    for epoch in range(config.flow_n_epochs):
        total_loss = 0.0
        
        # Shuffle pairs
        perm = torch.randperm(n_pairs, device=config.device)
        paired_z0_shuffled = paired_z0[perm]
        paired_z1_shuffled = paired_z1[perm]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * config.flow_batch_size
            end_idx = min(start_idx + config.flow_batch_size, n_pairs)
            
            z0_batch = paired_z0_shuffled[start_idx:end_idx]
            z1_batch = paired_z1_shuffled[start_idx:end_idx]
            
            # Compute loss
            optimizer.zero_grad()
            loss = flow_model.compute_cfm_loss(z0_batch, z1_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config.flow_n_epochs}, Loss: {avg_loss:.6f}")
    
    flow_model.vector_field.eval()
    print("Flow matching training complete!")
    
    return flow_model

