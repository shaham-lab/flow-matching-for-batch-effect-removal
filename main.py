"""Main pipeline for SFMBER batch correction."""

import argparse
import os
from anndata import AnnData
import scanpy as sc

from sfmber.data.load_data import load_adata, split_batches
from sfmber.data.preprocess import preprocess_data
from sfmber.models.scgen_wrapper import ScGenWrapper
from sfmber.models.flow_model import FlowMatchingModel
from sfmber.training.train_scgen import train_scgen
from sfmber.training.train_flow import train_flow_model
from sfmber.inference.batch_correct import correct_batch
from sfmber.evaluation.umap import plot_umap_comparison
from sfmber.utils.config import Config


def main():
    """Main pipeline for SFMBER."""
    parser = argparse.ArgumentParser(description="SFMBER: Stable Flow Matching for Batch Effect Removal")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data file")
    parser.add_argument("--batch_key", type=str, default="batch", help="Key for batch labels")
    parser.add_argument("--source_batch", type=str, default=None, help="Source batch label")
    parser.add_argument("--target_batch", type=str, default=None, help="Target batch label")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--scgen_epochs", type=int, default=100, help="scGen training epochs")
    parser.add_argument("--flow_epochs", type=int, default=100, help="Flow matching training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = Config(
        batch_key=args.batch_key,
        scgen_n_epochs=args.scgen_epochs,
        flow_n_epochs=args.flow_epochs,
        device=args.device
    )
    
    print("=" * 60)
    print("SFMBER: Stable Flow Matching for Batch Effect Removal")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    adata = load_adata(args.data_path)
    print(f"Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Step 2: Preprocess
    print("\n[2/6] Preprocessing data...")
    adata = preprocess_data(adata, n_top_genes=config.n_top_genes)
    print(f"After preprocessing: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Step 3: Split batches
    print("\n[3/6] Splitting batches...")
    source_adata, target_adata = split_batches(
        adata,
        batch_key=config.batch_key,
        source_batch=args.source_batch,
        target_batch=args.target_batch
    )
    print(f"Source batch: {source_adata.shape[0]} cells")
    print(f"Target batch: {target_adata.shape[0]} cells")
    
    # Combine for scGen training
    combined_adata = adata.copy()
    
    # Step 4: Train scGen
    print("\n[4/6] Training scGen VAE...")
    scgen_model = train_scgen(combined_adata, config)
    print("scGen training complete!")
    
    # Step 5: Train flow matching model
    print("\n[5/6] Training flow matching model...")
    flow_model = train_flow_model(
        source_adata,
        target_adata,
        scgen_model,
        config
    )
    print("Flow matching training complete!")
    
    # Step 6: Batch correction inference
    print("\n[6/6] Running batch correction...")
    corrected_adata = correct_batch(
        source_adata,
        scgen_model,
        flow_model,
        config
    )
    print("Batch correction complete!")
    
    # Save results
    output_path = os.path.join(args.output_dir, "corrected_data.h5ad")
    corrected_adata.write(output_path)
    print(f"\nSaved corrected data to {output_path}")
    
    # Create combined AnnData for visualization
    adata_before = source_adata.concatenate(target_adata, batch_key=config.batch_key)
    adata_after = corrected_adata.concatenate(target_adata, batch_key=config.batch_key)
    
    # Plot UMAP comparison
    print("\nGenerating UMAP visualization...")
    umap_path = os.path.join(args.output_dir, "umap_comparison.png")
    plot_umap_comparison(
        adata_before,
        adata_after,
        batch_key=config.batch_key,
        save_path=umap_path
    )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

