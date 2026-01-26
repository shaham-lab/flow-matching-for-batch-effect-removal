# SFMBER: Stable Flow Matching for Batch Effect Removal

A research-grade implementation of the SFMBER framework for correcting batch effects in single-cell RNA sequencing data using Conditional Flow Matching (CFM) within a learned latent manifold.

## Method Overview

SFMBER addresses batch correction by learning a time-dependent vector field in a latent space created by a Variational Autoencoder (VAE). The framework:

1. **Trains a VAE (scGen)** to map high-dimensional gene expression measurements into a shared latent space
2. **Computes Optimal Transport pairing** between source and target batches in latent space
3. **Learns a vector field** using Conditional Flow Matching that transports source batch distribution to target batch distribution
4. **Corrects batches** by integrating the learned vector field in latent space, then decoding back to expression space

This approach ensures that:
- Global distribution alignment matches the target batch
- Local biological identity and manifold structure are preserved
- Training is stable without "path crossing" phenomena

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd flow-matching-for-batch-effect-removal

# Install dependencies
pip install -r sfmber/requirements.txt
```

## Usage

### Basic Usage

```bash
python sfmber/main.py \
    --data_path /path/to/data.h5ad \
    --batch_key batch \
    --output_dir ./output
```

### Advanced Options

```bash
python sfmber/main.py \
    --data_path /path/to/data.h5ad \
    --batch_key batch \
    --source_batch batch_0 \
    --target_batch batch_1 \
    --output_dir ./output \
    --scgen_epochs 100 \
    --flow_epochs 100 \
    --device cpu
```

### Arguments

- `--data_path`: Path to input AnnData file (.h5ad format)
- `--batch_key`: Key in `adata.obs` containing batch labels (default: "batch")
- `--source_batch`: Label for source batch to correct (default: first batch)
- `--target_batch`: Label for target batch (default: second batch)
- `--output_dir`: Directory to save outputs (default: "./output")
- `--scgen_epochs`: Number of epochs for scGen training (default: 100)
- `--flow_epochs`: Number of epochs for flow matching training (default: 100)
- `--device`: Device to use - "cpu" or "cuda" (default: "cpu")

## Expected Inputs/Outputs

### Input

- **Format**: AnnData object (.h5ad file)
- **Required fields**:
  - `adata.X`: Gene expression matrix (cells × genes)
  - `adata.obs[batch_key]`: Batch labels for each cell

### Output

- **`corrected_data.h5ad`**: Corrected AnnData object with batch-corrected expression
- **`umap_comparison.png`**: UMAP visualization comparing before/after correction

## Project Structure

```
sfmber/
├── main.py                 # Main pipeline script
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── data/                  # Data loading and preprocessing
│   ├── load_data.py      # Data loading utilities
│   └── preprocess.py     # Preprocessing functions
│
├── models/                # Model definitions
│   ├── scgen_wrapper.py  # scGen VAE wrapper
│   ├── flow_model.py     # Flow matching model
│   └── ot_pairing.py     # Optimal Transport pairing
│
├── training/              # Training scripts
│   ├── train_scgen.py    # scGen training
│   ├── train_flow.py     # Flow matching training
│   └── losses.py         # Loss functions
│
├── inference/             # Inference modules
│   ├── integrate_flow.py # ODE integration
│   └── batch_correct.py  # Batch correction
│
├── evaluation/            # Evaluation utilities
│   └── umap.py           # UMAP visualization
│
└── utils/                 # Utilities
    └── config.py         # Configuration settings
```

## Key Components

### 1. Data Loading (`data/`)
- Loads AnnData objects using scanpy
- Splits data into source and target batches
- Preprocesses data (normalization, log transform, HVG selection)

### 2. scGen Wrapper (`models/scgen_wrapper.py`)
- Wraps scGen VAE to expose only encode/decode functionality
- Maps expression space ↔ latent space

### 3. Flow Matching Model (`models/flow_model.py`)
- Implements vector field MLP: v_θ(z, t)
- Uses linear interpolation: ψ_t(z₀, z₁) = (1-t)z₀ + t·z₁
- Trains using Conditional Flow Matching loss

### 4. Optimal Transport (`models/ot_pairing.py`)
- Computes empirical coupling π* using Sinkhorn algorithm
- Pairs source and target latent points for training

### 5. Training (`training/`)
- `train_scgen.py`: Trains VAE on combined batches
- `train_flow.py`: Trains flow model using OT-paired data

### 6. Inference (`inference/`)
- Integrates learned vector field using Euler/RK4
- Corrects batches in latent space, then decodes

## Technical Details

### Conditional Flow Matching Loss

The model minimizes:

```
L_CFM(θ) = E_{t ~ U(0,1), (z₀, z₁) ~ π*} ||v_θ(ψ_t(z₀, z₁), t) - (z₁ - z₀)||²
```

### Inference

At inference, batch correction is performed by:

1. Encoding: z₀ = E(x₀)
2. Integration: z_corr = z₀ + ∫[0 to 1] v_θ(z_τ, τ) dτ
3. Decoding: x_corr = D(z_corr)

## Dependencies

- `torch`: PyTorch for neural networks
- `scanpy`: Single-cell analysis
- `scgen`: VAE implementation
- `torchcfm`: Conditional Flow Matching
- `pot`: Python Optimal Transport
- `anndata`: Annotated data structures

## Citation

If you use SFMBER in your research, please cite:

```bibtex
@article{sfmber2024,
  title={Stable Flow Matching for Batch Effect Removal},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Specify your license here]

