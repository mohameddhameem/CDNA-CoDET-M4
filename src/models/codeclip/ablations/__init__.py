"""
CodeCLIP Ablation Study Variants

This directory contains intentional variations of the CodeCLIP model for
ablation studies and research purposes.

Files:
- CodeCLIP_ablation_graph_only.py
    Uses only graph modality, no text encoding
    Purpose: Measure contribution of graph information alone
    
- CodeCLIP_ablation_no_penalty.py
    Removes spectral regularization loss
    Purpose: Measure effect of covariance decorrelation penalty
    
- CodeCLIP_ablation_no_router.py
    Removes Mixture of Experts router in downstream task
    Purpose: Measure benefit of MoE fusion mechanism

For production use, import from src/models/codeclip/CodeCLIP.py instead.
"""
