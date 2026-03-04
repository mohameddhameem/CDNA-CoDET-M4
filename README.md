# LLM-4-SE
LLM for Software Engineering

This repository contains experiments and utilities for using large language models
to reason about and transform software systems as a "city of agents".

## Table of Contents
- [CodeCLIP Experiment Workflow](#codeclip-experiment-workflow) – End-to-end pretraining and evaluation
- [Project Structure](#project-structure) – Repository organization
- [Golden Sample Dataset](#golden-sample-dataset) – Dataset overview and usage

## CodeCLIP Experiment Workflow

Complete end-to-end workflow for running CodeCLIP pretraining and downstream tasks locally.

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# macOS/Linux:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (Optional) For GPU with CUDA 12.1
python -m pip install -r requirements-cu121.txt
```

### Step 2: Prepare Data

Organize your source code files and generate Code Property Graphs (CPGs):

```bash
# Your raw code should be organized in a directory, e.g., ./raw_code
# Then generate CPGs using Joern:
./scripts/generate_cpgs.sh

# This creates CPG GraphML files in ./CPG directory
```

### Step 3: Convert CPGs to Graph Format

Convert CPGs to heterogeneous or homogeneous graph representations:

```bash
# For heterogeneous graphs (node + edge types)
./scripts/generate_hetero.sh

# Or for homogeneous graphs (single node/edge type)
./scripts/generate_homo.sh

# Generated datasets are saved in ./CPG directory
```

### Step 4: Pretraining

Train the CodeCLIP model with contrastive learning on graph-code pairs:

```bash
# Basic pretraining (Python)
python run_experiments.py \
  --task_name pretrain \
  --language python \
  --epochs 100 \
  --batch_size 1024 \
  --use_gpu False

# With GPU
python run_experiments.py \
  --task_name pretrain \
  --language python \
  --epochs 100 \
  --use_gpu True \
  --gpu 0

# Or use shell script
./scripts/pretrain.sh
```

**Outputs**: Checkpoints saved to `./checkpoints/pretrain_*`

### Step 5: Downstream Task (Classification/Fine-tuning)

Fine-tune the pretrained model on downstream classification tasks:

```bash
# Run downstream task
python run_experiments.py \
  --task_name downstream \
  --language python \
  --use_gpu False

# Or use shell script
./scripts/downstream.sh
```

**Outputs**: Model predictions and metrics to `./results/`

### Step 6: Evaluation

Results are automatically computed during training:

```bash
# Check results
ls ./results/
cat ./results/downstream_*.log

# View metrics (precision, recall, F1, accuracy)
# Open notebooks for detailed analysis
jupyter notebook notebooks/exp_test.ipynb
```

### Available Command-Line Arguments

```bash
python run_experiments.py \
  --task_name pretrain              # pretrain or downstream
  --language python                 # python, java, or cpp
  --model samgpt                    # model name
  --batch_size 1024                 # batch size
  --learning_rate 0.001             # learning rate
  --epochs 100                      # number of epochs
  --hidden_dim 256                  # hidden dimension
  --use_gpu False                   # set True for GPU
  --checkpoints ./checkpoints       # checkpoint directory
```

See `run_experiments.py` for complete list of arguments.




## Project structure

### Source code (`src/`)
- **`models/codeclip/`** – CodeCLIP experiment models
  - `CodeCLIP.py`, `CodeBert.py`, `Unixcoder.py` (model architectures)
  - `GraphEncoder.py`, `TextEncoder.py` (core layers)
  - Ablation variants for research
- **`experiments/`** – Experiment orchestration
  - `exp_pretrain.py`, `exp_downstream.py` (training pipelines)
  - `exp_basic.py` (base class)
- **`data_processing/`** – Data utilities
  - `cpg2hetero.py`, `cpg2homo.py` (CPG conversions)
  - `tokenizer.py`, `datasplit.py`, `tools.py`
  - Legacy converters: `code_to_image_converter.py`, `clean_data.py`
- **`feature_extraction/`** – Feature extraction tools
- **`code_analysis/`** – Code analysis utilities (AST, graphs)
- **`city_building/`** – Visualization and simulation

### Experiment files (root level)
- **`run_experiments.py`** – main entry point for pretraining/downstream tasks
- **`scripts/`** – shell scripts for CPG generation and experiment execution
  - `pretrain.sh`, `downstream.sh`, `generate_hetero.sh`, `generate_homo.sh`, etc.

### Data
- **`data/raw/`** – original datasets (golden sample data)
- **`data/processed/`** – processed datasets and outputs

### Notebooks
- **`notebooks/`** – Jupyter notebooks for experimentation
  - `exp_test.ipynb` – CodeCLIP experiment testing
  - `Experiment_*.ipynb`, `demo*.ipynb` – other analysis notebooks

### Other
- **`requirements.txt`** – base dependencies (always install)
- **`requirements-cu121.txt`** – optional CUDA 12.1 extensions
- **`tests/`** – test and verification scripts

## Golden sample dataset

The `dataset/golden sample data` folder contains a **golden evaluation dataset**
built from Codeforces problems and multiple code-generation models. It is intended
for benchmarking and analysis of model‑written code versus human solutions.

### High-level summary

- **Problems**: 55 matched Codeforces problems (e.g. `1381/B Unmerge`, `1778/A Flip Flop Sum`)
- **Total samples**: 2,623 code samples
- **Models**:
  - `human` – accepted human submissions from Codeforces (via Kaggle)
  - `gpt` – GPT‑4o samples (from CoDeT‑M4 on HuggingFace)
  - `codellama` – CodeLlama‑7B samples (from CoDeT‑M4)
  - `llama3.1` – Llama‑3.1‑8B samples (from CoDeT‑M4 / raw JSON)
- **Languages**: `python`, `java`, `cpp`
- **Source datasets**: `DaniilOr/CoDET-M4` (HuggingFace) + raw JSON dumps from the original authors

The canonical machine‑readable overview of these counts and metadata is
`dataset/golden sample data/golden_sample_master.json`.

### Dataset folder layout

- **`dataset/golden sample data/code/`** – raw code files grouped by problem  
  - Layout: `code/<problem_id>/<model>_<language>.<ext>`  
  - Example: `code/1381_B/human_cpp.cpp`, `code/1381_B/gpt4o_python.py`

- **`dataset/golden sample data/golden_sample_code.jsonl`** – one row per code sample  
  Key fields:
  - `problem_id`, `contest_id`, `problem_index`, `problem_name`
  - `model`, `language`, `code`, `cleaned_code`
  - `code_file` (relative path under `code/`)
  - Static features such as `feature_avgFunctionLength`, `feature_maintainabilityIndex`,
    `loc`, `char_count`, `token_count_approx`

- **`dataset/golden sample data/golden_sample_metadata.csv`** – per‑sample CSV metadata  
  - Problem columns: `problem_id`, `contest_id`, `difficulty_rating`, `tags`, `solved_count`
  - Model / language columns: `model`, `model_display`, `language`, `source_type`, `hf_source`
  - Feature columns: `feature_*`, plus `loc`, `char_count`, `token_count_approx`, `code_file`

- **`dataset/golden sample data/golden_sample_problems.csv`** – per‑problem summary  
  - One row per Codeforces problem
  - Columns include: `problem_id`, `contest_id`, `problem_index`, `problem_name`,
    `difficulty_rating`, `tags`, `solved_count`, `codeforces_url`,
    `total_samples`, `models`, `languages`

- **`dataset/golden sample data/golden_sample_master.json`** – master summary file  
  - Top‑level keys: `description`, `models`, `model_descriptions`, `languages`,
    `source_dataset`, `matching_methods`, `total_problems`, `total_samples`
  - `problems`: array of per‑problem objects with `problem_id`, `problem_name`,
    `difficulty_rating`, `tags`, `codeforces_url`, `samples_count`,
    `models_present`, `languages_present`

- **Task guides** – markdown documentation about how the dataset was constructed  
  - `guide-task-1.6-1.7-treesitter.md`  
  - `guide-task-1.8-joern.md`  
  - `guide-task-1.11-visualization.md`

### Basic dataset usage examples

- **Python: iterate over all samples**

```python
import json
from pathlib import Path

root = Path("dataset") / "golden sample data"
jsonl_path = root / "golden_sample_code.jsonl"

with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        # row["problem_id"], row["model"], row["language"], row["code"], ...
```

- **Python: load per‑problem summary**

```python
import json
from pathlib import Path

master = json.loads(
    (Path("dataset") / "golden sample data" / "golden_sample_master.json")
        .read_text(encoding="utf-8")
)

print(master["total_problems"], master["total_samples"])
for problem in master["problems"]:
    print(problem["problem_id"], problem["problem_name"], problem["samples_count"])
```

