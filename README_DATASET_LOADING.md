# Dataset Loading: Complete Guide (Local + Colab)

## Quick Start

### For Google Colab (Recommended)

1. **Mount Drive** (Run this first):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Copy-Paste into Colab Cell**:
```python
exec(open('/content/drive/MyDrive/CPG/colab_quickstart.py').read())
```

3. **Configure** (edit at top of script):
```python
DATASET_PATH = "/content/drive/MyDrive/CPG"
FRACTION = 0.05  # Change to 0.1, 0.25, 0.5, or 1.0
BATCH_SIZE = 16
```

### For Local Machine

```bash
python load_dataset_configurable.py
```

Or in your Python code:
```python
from load_dataset_configurable import CPGDatasetManagerConfigurable
from torch.utils.data import DataLoader

manager = CPGDatasetManagerConfigurable()
dataset, metadata = manager.load_and_validate(fraction=0.05)
loader = DataLoader(dataset, batch_size=16)
```

---

## Files Available

| File | Purpose | Environment | Use Case |
|------|---------|-------------|----------|
| **colab_quickstart.py** | One-click Colab setup | Colab only | Immediate Colab usage |
| **load_dataset_configurable.py** | Main configurable loader | Both | Production code |
| **load_dataset_optimized.py** | Optimized loader (legacy) | Both | Advanced usage |
| **load_dataset.py** | Simple loader (legacy) | Both | Learning/reference |
| **CONFIGURATION_GUIDE.py** | All options explained | Both | Understanding options |
| **OPTIMIZATION_GUIDE.py** | Optimization techniques | Both | Performance tuning |

---

## Configuration Options

### Basic Parameters

```python
CPGDatasetManagerConfigurable(
    base_path=None,                          # Auto-detect or custom path
    processed_dir="processed_homo",          # or "processed_hetero"
    split_name="2way_base_split_target",     # JSON split file
    raw_jsonl="cpg_dataset_raw10.jsonl",     # JSONL file
    verbose=True,                            # Print details
    device="cpu"                             # or "cuda"
)
```

### Loading Data

```python
dataset, metadata = manager.load_and_validate(
    fraction=0.05,      # 0.01 to 1.0
    split="train"       # "train", "test", or "all"
)
```

### Fractions Explained

| Fraction | Samples | Memory | Time | Use Case |
|----------|---------|--------|------|----------|
| 0.01 | 2,400 | 0.3 GB | 15 sec | Ultra-quick test |
| 0.05 | 12,000 | 1.6 GB | 20 sec | **Recommended start** |
| 0.10 | 24,000 | 3.2 GB | 30 sec | Better validation |
| 0.25 | 60,000 | 8 GB | 1-2 min | Good coverage |
| 0.50 | 120,000 | 16 GB | 2-3 min | Thorough test |
| 1.0 | 240,000 | 32 GB | 5+ min | Full production |

---

## Example: Full Training Loop

### Local Machine
```python
from load_dataset_configurable import CPGDatasetManagerConfigurable
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Initialize
manager = CPGDatasetManagerConfigurable(device="cpu")

# Load dataset (5% for testing)
dataset, meta = manager.load_and_validate(fraction=0.05)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Training with {len(dataset)} samples")
print(f"Classes: {meta['label_distribution']}")

# Training loop
for epoch in range(10):
    for batch in loader:
        graph, label, idx, name = batch

        # Your model forward pass
        # output = model(graph)
        # loss = F.cross_entropy(output, label)
        # loss.backward()
        # optimizer.step()

        print(f"Batch processed: {len(label)} samples")
        break  # Just show first batch
```

### Google Colab
```python
# In Colab cell 1: Mount drive
from google.colab import drive
drive.mount('/content/drive')

# In Colab cell 2: Load dataset
from pathlib import Path

DATASET_PATH = "/content/drive/MyDrive/CPG"
FRACTION = 0.05

exec(open('/content/drive/MyDrive/CPG/colab_quickstart.py').read())

# Now 'loader', 'metadata', 'dataset' are available
for batch in loader:
    graph, label, idx, name = batch
    print(f"Got batch with {len(label)} samples")
    break
```

---

## Auto-Detection: How It Works

The loader automatically detects your environment:

```python
from load_dataset_configurable import EnvironmentDetector

EnvironmentDetector.print_environment_info()
```

**Output for Local:**
```
Running in: Local Machine
Default path: C:\Learning\SMU\City-of-Agents-1\dataset
```

**Output for Colab:**
```
Running in: Google Colab
Default path: /content/drive/MyDrive/CPG
```

---

## Custom Paths

### Local (Windows)
```python
manager = CPGDatasetManagerConfigurable(
    base_path=r"C:\Your\Custom\Path\dataset"
)
```

### Local (Linux/Mac)
```python
manager = CPGDatasetManagerConfigurable(
    base_path="/home/user/your_dataset"
)
```

### Colab
```python
manager = CPGDatasetManagerConfigurable(
    base_path="/content/drive/MyDrive/YourFolder"
)
```

---

## Directory Structure Expected

```
your_dataset_folder/
├── data_split/
│   ├── 2way_base_split_target.json
│   ├── 2way_base_split_model.json
│   ├── 2way_50tasks_target.json
│   └── 7way_base_split_model.json
├── processed_homo/
│   ├── processed.pt          (32 GB)
│   ├── pre_filter.pt
│   └── pre_transform.pt
├── processed_hetero/
│   ├── processed.pt
│   ├── pre_filter.pt
│   └── pre_transform.pt
└── cpg/ (or raw10/)
    └── cpg_dataset_raw10.jsonl
```

---

## Memory Usage Estimation

For different fractions:

```
Metadata (always):     ~200 MB
  - Labels for all samples
  - Split indices

Graphs (loaded on demand):
  - 1%:   0.3 GB
  - 5%:   1.6 GB
  - 10%:  3.2 GB
  - 25%:  8 GB
  - 50%:  16 GB
  - 100%: 32 GB (full)
```

### RAM Recommendations
- **8 GB RAM**: Use 1-2% fractions with sequential loading
- **16 GB RAM**: Use 5% fractions with batch_size=8-16
- **32+ GB RAM**: Can handle up to 50% fractions

---

## Troubleshooting

### Issue: FileNotFoundError
**Problem**: Paths not found
**Solution**:
```python
# Check what's available
from pathlib import Path
dataset_path = Path("/content/drive/MyDrive/CPG")
for item in dataset_path.iterdir():
    print(item)

# Use correct base_path
manager = CPGDatasetManagerConfigurable(base_path=str(dataset_path))
```

### Issue: MemoryError
**Problem**: Loading too much data
**Solution**: Use smaller fraction
```python
# Instead of 0.25, use 0.05
dataset, meta = manager.load_and_validate(fraction=0.05)
```

### Issue: Slow loading
**Problem**: Loading entire JSONL
**Solution**: Use metadata-only mode first
```python
manager.load_split_data()  # Fast: 1 second
manager.load_labels_from_jsonl()  # Slower: 10-15 seconds
```

### Issue: CUDA not available
**Problem**: GPU device errors
**Solution**: Use CPU
```python
manager = CPGDatasetManagerConfigurable(device="cpu")
```

---

## Advanced: Sequential Loading

For limited memory, process subsets sequentially:

```python
manager = CPGDatasetManagerConfigurable()
manager.load_split_data()
manager.load_labels_from_jsonl()

# Process 10 subsets of 5% each
for i in range(10):
    indices = manager.create_subset_indices(fraction=0.05)
    dataset = manager.create_dataset(indices)

    # Process dataset
    loader = DataLoader(dataset, batch_size=16)
    for batch in loader:
        # Do something
        pass

    # Cleanup
    del dataset, loader
    import gc
    gc.collect()
```

**Memory**: Never exceeds ~2 GB

---

## Performance Comparison

```
Quick Test (5%):
  Load time: 20 seconds
  Memory: 1.6 GB
  Batches: 750

Good Validation (25%):
  Load time: 1-2 minutes
  Memory: 8 GB
  Batches: 3,750

Full Training (100%):
  Load time: 5+ minutes
  Memory: 32 GB
  Batches: 15,000
```

---

## Environment Variables (Optional)

Set these before importing if needed:

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'  # GPU memory
```

---

## API Reference

### CPGDatasetManagerConfigurable

**Methods:**
- `load_split_data()` - Load JSON split file
- `load_labels_from_jsonl()` - Load labels from JSONL
- `create_subset_indices(fraction, split, stratified)` - Create subset
- `create_dataset(indices, load_mode)` - Create dataset from indices
- `load_and_validate(fraction, split)` - Complete pipeline

**Properties:**
- `split_data` - Loaded split JSON
- `labels` - Loaded labels dict
- `label_map` - Label name to index
- `id_map` - Index to label name

### CPGDatasetOptimizedConfigurable

**Methods:**
- `__len__()` - Dataset size
- `__getitem__(idx)` - Get single sample
- `label_distribution` - Class balance

**Returns from `__getitem__`:**
```python
graph, label, graph_idx, label_name = dataset[0]
```

---

## Getting Help

1. **Configuration issues**: See `CONFIGURATION_GUIDE.py`
2. **Optimization tips**: See `OPTIMIZATION_GUIDE.py`
3. **Quick start**: Run `colab_quickstart.py` or `load_dataset_configurable.py`

---

## Summary

| Use Case | Command |
|----------|---------|
| **Colab (quick)** | Paste `colab_quickstart.py` into cell |
| **Colab (custom)** | Edit config at top + run |
| **Local (quick)** | `python load_dataset_configurable.py` |
| **Local (custom)** | Use `CPGDatasetManagerConfigurable` class |
| **Production** | Use `fraction=1.0` with appropriate device |

---

**Last Updated**: Feb 2025
**Compatible With**: PyTorch 2.0+, Python 3.8+
