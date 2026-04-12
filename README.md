# CDNA-CoDET-M4

Code authorship attribution using code property graphs and graph-text models.

This project extends the original [CoDET-M4 dataset](https://huggingface.co/datasets/DaniilOr/CoDET-M4) 
by enriching it with **Code Property Graph (CPG)** representations extracted via Joern static analysis. 
The enhanced dataset is published as **CDNA-CoDET-M4** on HuggingFace and used for training 
graph-based neural models to attribute code authorship across multiple LLMs.

## Dataset

**CDNA-CoDET-M4** is available on HuggingFace:  
https://huggingface.co/datasets/mohameddhameem/CDNA-CoDET-M4

This is an enhanced version of the original CoDET-M4 dataset with two configurations:

- **`hetero`**: Full heterogeneous Code Property Graphs (33K+ samples, ~11 GB)
  - Node and edge structures from AST, CFG, PDG
  - GraphML serialized format
  - Best for graph neural network training

- **`scalar`**: Aggregated structural metrics (77K+ samples, ~26 GB)
  - 22 precomputed graph features (complexity, density, fan-in/fan-out, etc.)
  - Tabular format suitable for traditional ML
  - Better for interpretability studies

Load the dataset:

```python
from datasets import load_dataset

# Heterogeneous graphs for GNN training
hetero = load_dataset("mohameddhameem/CDNA-CoDET-M4", "hetero")

# Scalar features for baseline models
scalar = load_dataset("mohameddhameem/CDNA-CoDET-M4", "scalar")
```

## Workflow

1. **Source**: Original CoDET-M4 dataset (~150K samples, 6 LLM models + human)
2. **CPG Extraction**: Joern static analyzer converts code to control/data flow graphs
3. **Heterogeneous Graph**: Multi-edge-type graph combining AST, CFG, PDG
4. **Publishing**: Enhanced dataset uploaded to HuggingFace as CDNA-CoDET-M4
5. **Training**: Graph-text models (CodeCLIP-style) for code authorship classification

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Optional GPU acceleration (CUDA 12.1):

```bash
pip install -r requirements-cu121.txt
```

## Quick Start

### Download & Process Dataset

To use the prepared CDNA-CoDET-M4 dataset (recommended):

```bash
# Will auto-download and cache from HuggingFace
python run.py --task_name pretrain --model full --train_language all --path CPG
```

To generate CPGs from raw code (requires Joern):

```bash
bash scripts/generate_cpgs.sh
```

### Pretrain (Unsupervised)

Train graph-text encoder with contrastive loss:

```bash
python run.py --task_name pretrain \
  --model full \
  --train_language all \
  --path CPG \
  --epochs 100 \
  --batch_size 64
```

### Downstream (Supervised Classification)

Fine-tune for code authorship attribution:

```bash
python run.py --task_name downstream \
  --model full \
  --pattern pretrain \
  --train_language python \
  --test_languages python,java \
  --path CPG
```

See `scripts/` for shell wrappers with full hyperparameters.

## Models

- **`full`**: Graph encoder + text encoder (CodeBERT) with contrastive loss
- **`codebert`**: Text-only baseline (CodeBERT)
- **`graph_only`**: Graph-only ablation (no text)
- **`no_cross`**, **`no_penalty`**, **`no_router`**: Loss function ablations

## Project Structure

- `run.py`: Main CLI entry point
- `exp/`: Training loops (pretrain, downstream, evaluation)
- `models/`: Model variants, ablations
- `layers/`: Heterogeneous graph and text encoders
- `utils/`: Data conversion (Joern, CPG parsing, tokenization)
- `scripts/`: Batch processing and shell helpers
- `CPG/`: Dataset root (auto-populated after first run)

## Citation

If you use this dataset or code, please cite:

```bibtex
@dataset{cdna_codetm4,
  title   = {{CDNA-CoDET-M4}: Code Authorship Attribution via Code Property Graphs},
  authors = {Gusta, Avisenna and Yinqi, Gu and Sia, Sim Kim and Mohamed, Dhameem and Shenghua, Ye},
  year    = {2025},
  url     = {https://huggingface.co/datasets/mohameddhameem/CDNA-CoDET-M4}
}
```

Original CoDET-M4 reference: https://huggingface.co/datasets/DaniilOr/CoDET-M4

