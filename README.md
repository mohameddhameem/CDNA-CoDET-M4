# LLM-4-SE
LLM for Software Engineering

This repository contains experiments and utilities for using large language models
to reason about and transform software systems as a "city of agents".

## Quick start

1. **Create and activate a virtual environment** (recommended):
   - `python -m venv .venv`
   - macOS/Linux: `source .venv/bin/activate`
   - Windows: `.\.venv\Scripts\activate`
2. **Install base dependencies** (always):
   - `python -m pip install --upgrade pip`
   - `python -m pip install -r requirements.txt`
3. **Optional: install CUDA 12.1 PyG extensions** (Linux/Windows GPU only):
   - `python -m pip install -r requirements-cu121.txt`
4. **Run an example pipeline** (from the repo root):
   - `python build_city.py`
   - or open one of the notebooks under `notebooks/` in Jupyter / VS Code.

## Project structure

- **Root Python scripts**: core utilities and pipelines for the project  
  - `build_city.py`, `build_city_buildings.py`, `build_city_research.py`, `build_city_v2.py`  
  - `clean_data.py`, `generate_simluation.py`, `universal_parser.py`  
  - Graph/feature utilities: `ast2pyg.py`, `pyg_creator.py`, `create_feature.py`, `CodeCLIP.py`, `Joern.py`

- **`notebooks/`**: interactive experimentation and demos  
  - `Experiment_1.ipynb`  
  - `demo2.ipynb`  
  - `demo6.ipynb`

- **`outputs/`**: generated artifacts and data dumps  
  - City visualizations: `city_map*.html`  
  - Simulation data: `simulation_data.csv`

- **`scripts/`**: helper shell scripts  
  - `scripts/generate_graphml.sh`

- **`requirements.txt`**: base dependencies for data/modeling and CPU-friendly setup
- **`requirements-cu121.txt`**: optional CUDA 12.1 + PyG extension packages

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

