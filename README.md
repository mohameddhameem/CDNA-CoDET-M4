# City of Agents

A framework for analyzing code structure, generating graph representations, and visualizing software metrics.

## Quick Start

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohameddhameem/City-of-Agents-1.git
   cd City-of-Agents-1
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install the package in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Run an example**:
   ```bash
   # Generate sample data
   python -m city_of_agents.simulation.generate_simulation
   
   # Build visualization
   python -m city_of_agents.builders.city
   ```

### Google Colab Setup

1. Open the **Colab Setup Notebook**: [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb)
2. Run all cells to clone, install, and verify the setup

## Project Structure

```
City-of-Agents-1/
├── src/
│   └── city_of_agents/          # Main package
│       ├── __init__.py
│       ├── ast2pyg.py            # AST to PyG Data converter
│       ├── joern.py              # Joern CPG runner
│       ├── codeclip.py           # Code-Graph CLIP models
│       ├── parsers/              # Code parsing utilities
│       │   ├── __init__.py
│       │   └── universal_parser.py
│       ├── builders/             # City visualization builders
│       │   ├── __init__.py
│       │   ├── city.py           # Basic pillars visualization
│       │   ├── city_v2.py        # Lollipop style
│       │   ├── city_buildings.py # 3D buildings
│       │   └── city_research.py  # Research edition with metrics
│       ├── utils/                # Data processing utilities
│       │   ├── __init__.py
│       │   ├── clean_data.py
│       │   ├── create_feature.py
│       │   └── pyg_creator.py
│       └── simulation/           # Simulation generators
│           ├── __init__.py
│           └── generate_simulation.py
│
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── Colab_Setup.ipynb        # Google Colab setup guide
│   ├── Experiment_1.ipynb
│   ├── Experiment_Models.ipynb
│   ├── demo2.ipynb
│   └── demo6.ipynb
│
├── dataset/                      # Golden sample dataset
│   └── golden sample data/
│       ├── code/                 # Code samples by problem
│       ├── golden_sample_code.jsonl
│       ├── golden_sample_master.json
│       ├── golden_sample_metadata.csv
│       └── golden_sample_problems.csv
│
├── outputs/                      # Generated visualizations and data
│   ├── city_map*.html
│   └── simulation_data.csv
│
├── scripts/                      # Helper scripts
│   └── generate_graphml.sh
│
├── setup.py                      # Package installation script
├── pyproject.toml               # Modern Python packaging config
├── requirements.txt             # Dependencies
├── requirements-dev.txt         # Development dependencies
└── README.md
```

## Installation Options

### Option 1: Editable Install (Recommended for Development)

For development with immediate code updates:
```bash
pip install -e .
```

For production:
```bash
pip install .
```

With development dependencies:
```bash
pip install -e ".[dev]"
```

## Usage Examples

### 1. Generate Simulation Data

```python
from city_of_agents.simulation import generate_simulation

# Generates simulation_data.csv with synthetic code metrics
```

### 2. Build City Visualizations

```python
from city_of_agents.builders import city

# Creates city_map.html with 3D visualization
```

### 3. Convert Python AST to PyG Data

```python
from city_of_agents.ast2pyg import ast_to_pyg_data

code = """
def hello_world():
    print("Hello, world!")
"""

data = ast_to_pyg_data(code, target_label=0)
print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
```

### 4. Use Joern for CPG Generation

```python
from city_of_agents.joern import JoernRunner

runner = JoernRunner(output_dir="./CPG")
graphml_path = runner.parse_string(code_string, language="python")
```

### 5. Create PyG Dataset

```python
from city_of_agents.utils.pyg_creator import create_cpg_dataset

dataset = create_cpg_dataset("./CPG", force_reload=False)
print(f"Dataset size: {len(dataset)}")
```

## Visualization Styles

The framework supports multiple visualization approaches:

- `city.py`: Basic pillars
- `city_v2.py`: Lollipop-style with floor shadows
- `city_buildings.py`: 3D buildings
- `city_research.py`: Research metrics edition

All output interactive HTML files.

## Golden Sample Dataset

The `dataset/golden sample data` folder contains an evaluation dataset from Codeforces problems with multiple code-generation models.

### Dataset Summary

- 55 Codeforces problems
- 2,623 total code samples
- Multiple models: Human, GPT-4o, CodeLlama-7B, Llama-3.1-8B
- Languages: Python, Java, C++
- Source: `DaniilOr/CoDET-M4` (HuggingFace)

### Dataset Files

| File | Description |
|------|-------------|
| `golden_sample_code.jsonl` | Per-sample data with code and metrics |
| `golden_sample_master.json` | Master summary with problem metadata |
| `golden_sample_metadata.csv` | CSV format metadata |
| `golden_sample_problems.csv` | Per-problem summary |
| `code/` | Raw code files organized by problem |

### Usage Example

```python
import json
from pathlib import Path

jsonl_path = Path("dataset/golden sample data/golden_sample_code.jsonl")
with jsonl_path.open("r") as f:
    for line in f:
        sample = json.loads(line)
        print(f"{sample['problem_id']}: {sample['model']} ({sample['language']})")
```

## Running Experiments

### Local Environment

```bash
# 1. Generate simulation data
python -m city_of_agents.simulation.generate_simulation

# 2. Build a visualization
python -m city_of_agents.builders.city

# 3. Open the generated HTML file
# Windows: outputs/city_map.html
# Linux/Mac: open outputs/city_map.html
```

### Jupyter Notebooks

```bash
jupyter notebook
```

Then open any notebook in the `notebooks/` folder.

### Google Colab

Upload [colab_setup.ipynb](notebooks/colab_setup.ipynb) and run all cells.

## Advanced Features

### PyTorch Geometric Integration

Convert code ASTs to PyG Data objects for GNN training:

```python
from city_of_agents.ast2pyg import ast_to_pyg_data
from torch_geometric.loader import DataLoader

# Convert multiple code samples
data_list = [ast_to_pyg_data(code, label) for code, label in zip(codes, labels)]

# Create PyG DataLoader
loader = DataLoader(data_list, batch_size=32, shuffle=True)
```

### Feature Encoding

Encode node/edge texts using transformer models:

```python
from city_of_agents.utils.create_feature import encode_text_graph

# Encode using sentence transformers
dataset = encode_text_graph(dataset, lm_type="tiny", batch_size=32)
```

### Code CLIP Models

Train contrastive models on code graphs and text:

```python
from city_of_agents.codeclip import Pretrain

model = Pretrain(graph_in_dim=384, text_vocab_size=30522, embed_dim=256)
logits_graph, logits_text = model(graph_batch, text_input)
loss = model.loss(logits_graph, logits_text)
```

## Contributing

Fork the repository, create a feature branch, and open a pull request.

## License

MIT License

## Acknowledgments

- Dataset: `DaniilOr/CoDET-M4` (HuggingFace)
- Visualization: Plotly
- Graph Processing: PyTorch Geometric
- Code Analysis: Joern

