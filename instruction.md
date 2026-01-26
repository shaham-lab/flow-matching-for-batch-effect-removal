Build a full Python project implementing the SFMBER (Stable Flow Matching for Batch Effect Removal) pipeline using:

- scGen (https://github.com/theislab/scgen) as the VAE encoder/decoder
- Conditional Flow Matching from https://github.com/atong01/conditional-flow-matching (torchcfm)
- scanpy / AnnData for dataset loading and preprocessing

The code must be clean, modular, and research-grade.

================================================
PROJECT STRUCTURE
================================================

Create the following structure and generate ALL files:

sfmber/
├── main.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── load_data.py
│   └── preprocess.py
│
├── models/
│   ├── scgen_wrapper.py
│   ├── flow_model.py
│   └── ot_pairing.py
│
├── training/
│   ├── train_scgen.py
│   ├── train_flow.py
│   └── losses.py
│
├── inference/
│   ├── integrate_flow.py
│   └── batch_correct.py
│
├── evaluation/
│   └── umap.py
│
└── utils/
    └── config.py

================================================
FUNCTIONAL REQUIREMENTS
================================================

1. DATA LOADING (scanpy)
- Use scanpy.read() to load AnnData
- Split dataset into source batch B0 and target batch B1 using a batch key
- Preprocess data (normalization, log1p, HVGs)

2. SCGEN WRAPPER
- Wrap scGen into a class ScGenWrapper
- Expose:
    - encode(adata) -> torch.Tensor
    - decode(z) -> numpy array
- Hide scGen internal training logic

3. FLOW MATCHING (torchcfm)
- Use ExactOptimalTransportConditionalFlowMatcher from torchcfm
- Implement latent-space conditional flow matching
- Use linear interpolation ψ_t = (1−t)z0 + tz1
- Vector field v_theta(z, t) implemented as an MLP

4. TRAINING
- train_scgen.py:
    - Train scGen on combined batches
- train_flow.py:
    - Encode B0 and B1
    - Compute OT pairing
    - Train flow model using CFM loss

5. INFERENCE
- Integrate the learned vector field using Euler or RK4
- Correct B0 → B1 in latent space
- Decode corrected latent to expression space

6. MAIN PIPELINE (main.py)
main.py must:
- Load dataset using scanpy
- Train scGen
- Train flow model
- Run batch correction inference
- Save corrected AnnData
- Plot UMAP before/after correction

================================================
STYLE REQUIREMENTS
================================================

- Use PyTorch throughout
- Add docstrings and type hints
- Keep code minimal but complete
- No placeholders — code must run
- Use CPU by default
- Avoid unnecessary abstractions

================================================
DELIVERABLES
================================================

- Full runnable code
- requirements.txt
- README explaining:
    - Method overview
    - How to run training + inference
    - Expected inputs/outputs

================================================
IMPORTANT
================================================

- Flow matching MUST use torchcfm (do NOT reimplement CFM from scratch)
- scGen must ONLY be used as a VAE (latent encoder/decoder)
- Batch correction must occur ONLY in latent space

Generate all files now.
