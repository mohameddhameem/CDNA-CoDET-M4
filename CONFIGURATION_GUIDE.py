"""
CONFIGURATION GUIDE: All Options and Examples
"""

CONFIG_GUIDE = """
╔════════════════════════════════════════════════════════════════════════════╗
║              DATASET LOADER: CONFIGURATION OPTIONS                        ║
║                    (Local + Colab Compatible)                             ║
╚════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
1. ENVIRONMENT DETECTION (Automatic)
═══════════════════════════════════════════════════════════════════════════════

The loader automatically detects your environment and sets default paths:

LOCAL MACHINE:
  Default path: C:\\Learning\\SMU\\City-of-Agents-1\\dataset
  Detected by: Checking for google.colab import
  Override: Pass custom base_path parameter

GOOGLE COLAB:
  Default path: /content/drive/MyDrive/CPG
  Detected by: google.colab module present
  Override: Pass custom base_path parameter


═══════════════════════════════════════════════════════════════════════════════
2. INITIALIZATION OPTIONS
═══════════════════════════════════════════════════════════════════════════════

Basic (Auto-detect everything):
────────────────────────────────────────────────────────────────────────────
  from load_dataset_configurable import CPGDatasetManagerConfigurable

  manager = CPGDatasetManagerConfigurable()
  # Uses defaults for your environment

Custom path (Local):
────────────────────────────────────────────────────────────────────────────
  manager = CPGDatasetManagerConfigurable(
      base_path=r"C:\\Your\\Custom\\Path\\dataset"
  )

Custom path (Colab):
────────────────────────────────────────────────────────────────────────────
  manager = CPGDatasetManagerConfigurable(
      base_path="/content/drive/MyDrive/YourFolder"
  )

Full configuration:
────────────────────────────────────────────────────────────────────────────
  manager = CPGDatasetManagerConfigurable(
      base_path=None,                          # Auto-detect
      processed_dir="processed_homo",          # or "processed_hetero"
      split_name="2way_base_split_target",     # JSON split file
      raw_jsonl="cpg_dataset_raw10.jsonl",     # JSONL file
      verbose=True,                            # Print details
      device="cpu"                             # or "cuda"
  )


═══════════════════════════════════════════════════════════════════════════════
3. PROCESSED DIRECTORY OPTIONS
═══════════════════════════════════════════════════════════════════════════════

Homogeneous Graphs (default):
────────────────────────────────────────────────────────────────────────────
  processed_dir="processed_homo"
  Contents: Single type of nodes/edges
  Use when: Building models for homogeneous graphs

Heterogeneous Graphs:
────────────────────────────────────────────────────────────────────────────
  processed_dir="processed_hetero"
  Contents: Multiple node/edge types
  Use when: Working with heterogeneous graph neural networks


═══════════════════════════════════════════════════════════════════════════════
4. SPLIT FILE OPTIONS
═══════════════════════════════════════════════════════════════════════════════

Binary Classification (2-way):
────────────────────────────────────────────────────────────────────────────
  split_name="2way_base_split_target"
  Classes: AI (0) vs Human (1)
  Use when: Binary human vs AI classification

Multi-class Classification (7-way):
────────────────────────────────────────────────────────────────────────────
  split_name="7way_base_split_model"
  Classes: Multiple AI models
  Use when: Classifying which AI model generated code


═══════════════════════════════════════════════════════════════════════════════
5. LOADING DATA OPTIONS
═══════════════════════════════════════════════════════════════════════════════

Small Fraction (5% - Quick Testing):
────────────────────────────────────────────────────────────────────────────
  dataset, meta = manager.load_and_validate(fraction=0.05)

  Memory: ~1.6 GB
  Time: ~30 seconds
  Use: Quick prototyping, debugging

Medium Fraction (25% - Validation):
────────────────────────────────────────────────────────────────────────────
  dataset, meta = manager.load_and_validate(fraction=0.25)

  Memory: ~8 GB
  Time: ~1-2 minutes
  Use: Testing model performance

Large Fraction (50% - Good Coverage):
────────────────────────────────────────────────────────────────────────────
  dataset, meta = manager.load_and_validate(fraction=0.50)

  Memory: ~16 GB
  Time: ~2-3 minutes
  Use: Representative testing

Full Dataset (100% - Production):
────────────────────────────────────────────────────────────────────────────
  dataset, meta = manager.load_and_validate(fraction=1.0)

  Memory: ~32 GB
  Time: ~5+ minutes
  Use: Full model training


═══════════════════════════════════════════════════════════════════════════════
6. SPLIT OPTIONS (Train/Test/All)
═══════════════════════════════════════════════════════════════════════════════

Training Set:
────────────────────────────────────────────────────────────────────────────
  dataset, meta = manager.load_and_validate(split="train", fraction=0.05)
  Samples: ~190,000 (full) / ~9,500 (5%)

Test Set:
────────────────────────────────────────────────────────────────────────────
  dataset, meta = manager.load_and_validate(split="test", fraction=0.05)
  Samples: ~50,000 (full) / ~2,500 (5%)

All Data:
────────────────────────────────────────────────────────────────────────────
  dataset, meta = manager.load_and_validate(split="all", fraction=0.05)
  Samples: ~240,000 (full) / ~12,000 (5%)


═══════════════════════════════════════════════════════════════════════════════
7. DEVICE OPTIONS
═══════════════════════════════════════════════════════════════════════════════

CPU (Default):
────────────────────────────────────────────────────────────────────────────
  manager = CPGDatasetManagerConfigurable(device="cpu")
  Use when: Testing or limited GPU

GPU (CUDA):
────────────────────────────────────────────────────────────────────────────
  manager = CPGDatasetManagerConfigurable(device="cuda")
  Use when: Training with GPU acceleration
  Note: Make sure torch is compiled with CUDA support


═══════════════════════════════════════════════════════════════════════════════
8. USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Quick Test on Local Machine
────────────────────────────────────────────────────────────────────────────
from load_dataset_configurable import CPGDatasetManagerConfigurable
from torch.utils.data import DataLoader

manager = CPGDatasetManagerConfigurable(verbose=True)
dataset, meta = manager.load_and_validate(fraction=0.05, split="train")
loader = DataLoader(dataset, batch_size=16)

for batch in loader:
    graph, label, idx, name = batch
    print(f"Batch: {label}")
    break


EXAMPLE 2: Colab with Custom Path
────────────────────────────────────────────────────────────────────────────
manager = CPGDatasetManagerConfigurable(
    base_path="/content/drive/MyDrive/CPG",
    processed_dir="processed_homo",
    split_name="2way_base_split_target"
)
dataset, meta = manager.load_and_validate(fraction=0.1)


EXAMPLE 3: Production Training (Full Dataset)
────────────────────────────────────────────────────────────────────────────
manager = CPGDatasetManagerConfigurable(device="cuda")
dataset, meta = manager.load_and_validate(fraction=1.0, split="train")
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Full training loop
for epoch in range(num_epochs):
    for batch in loader:
        # Your training code


EXAMPLE 4: Multiple Subsets (Memory Efficient)
────────────────────────────────────────────────────────────────────────────
manager = CPGDatasetManagerConfigurable()
manager.load_split_data()
manager.load_labels_from_jsonl()

# Process 10 subsets, each 5%
for i in range(10):
    indices = manager.create_subset_indices(fraction=0.05)
    dataset = manager.create_dataset(indices)

    # Process this subset
    # Memory never exceeds ~2GB

    del dataset


═══════════════════════════════════════════════════════════════════════════════
9. COMMAND-LINE USAGE (Colab Cell)
═══════════════════════════════════════════════════════════════════════════════

Paste this entire code into a Colab cell:

from colab_quickstart import load_dataset_colab

loader, meta, dataset = load_dataset_colab(
    dataset_path="/content/drive/MyDrive/CPG",
    processed_dir="processed_homo",
    split_name="2way_base_split_target",
    fraction=0.05,
    batch_size=16,
    device="cpu"
)

# Use loader in training
for batch in loader:
    graph, label, idx, name = batch
    break


═══════════════════════════════════════════════════════════════════════════════
10. TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

Problem: FileNotFoundError: processed.pt not found
Solution: Check processed_dir parameter
  manager = CPGDatasetManagerConfigurable(processed_dir="processed_hetero")

Problem: FileNotFoundError: 2way_base_split_target.json not found
Solution: Check split_name parameter or list available splits
  split_name="2way_base_split_model"  # Try different split

Problem: Memory Error when loading
Solution: Reduce fraction
  fraction=0.01  # Instead of 0.05

Problem: "Which device to use?"
Solution: Check availability
  # CPU is always available
  # GPU only if torch compiled with CUDA
  device="cpu"  # Safe default

Problem: Colab connection timeout
Solution: Reduce fraction or increase timeout
  fraction=0.02  # Very small subset
  # Or split into sequential loads


═══════════════════════════════════════════════════════════════════════════════
11. QUICK REFERENCE COMMANDS
═══════════════════════════════════════════════════════════════════════════════

LOCAL:
  python load_dataset_configurable.py

COLAB:
  from colab_quickstart import load_dataset_colab
  loader, meta, dataset = load_dataset_colab(...)

SHOW ALL STRATEGIES:
  python dataset_loading_strategies.py

GET ENVIRONMENT INFO:
  from load_dataset_configurable import EnvironmentDetector
  EnvironmentDetector.print_environment_info()

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(CONFIG_GUIDE)

    # Save to file
    with open(r"c:\Learning\SMU\City-of-Agents-1\CONFIGURATION_GUIDE.txt", 'w') as f:
        f.write(CONFIG_GUIDE)

    print("\n✓ Configuration guide saved!")
