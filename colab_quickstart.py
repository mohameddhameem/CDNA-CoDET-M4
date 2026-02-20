"""
COLAB QUICKSTART
Copy-paste this entire cell into Google Colab
Works with: /content/drive/MyDrive/CPG
Auto-detects GPU if available
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION (Edit these variables)
# ============================================================================
DATASET_PATH = "/content/drive/MyDrive/CPG"
PROCESSED_DIR = "processed_homo"
SPLIT_NAME = "2way_base_split_target"
FRACTION = 0.05
BATCH_SIZE = 16

# DEVICE: 'auto' (recommended), 'cuda' (force GPU), or 'cpu' (force CPU)
DEVICE = "auto"
# ============================================================================


class DeviceManager:
    """Simple device detection"""

    @staticmethod
    def is_cuda_available():
        return torch.cuda.is_available()

    @staticmethod
    def auto_select():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def validate(device_str):
        if device_str == "auto":
            return DeviceManager.auto_select()
        if device_str == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available. Using CPU.")
            return "cpu"
        return device_str if device_str in ["cuda", "cpu"] else "cpu"


class CPGDatasetColab(Dataset):
    """Simple dataset for Colab"""

    def __init__(self, tensor_path, indices, labels_dict, id_map, device="cpu"):
        self.tensor_path = tensor_path
        self.indices = indices
        self.labels_dict = labels_dict
        self.id_map = id_map
        self.device = device
        self._graphs = None

    def _load_graphs(self):
        if self._graphs is None:
            self._graphs = torch.load(self.tensor_path, map_location=self.device)

    @property
    def graphs(self):
        self._load_graphs()
        return self._graphs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._load_graphs()
        graph_idx = self.indices[idx]
        graph = self._graphs[graph_idx]
        label = self.labels_dict.get(graph_idx, -1)
        label_name = self.id_map.get(str(label), 'unknown')
        return graph, label, graph_idx, label_name

    @property
    def label_distribution(self):
        labels = [self.labels_dict.get(idx, -1) for idx in self.indices]
        unique, counts = np.unique(labels, return_counts=True)
        dist = {}
        for label_idx, count in zip(unique, counts):
            label_name = self.id_map.get(str(label_idx), 'unknown')
            dist[label_name] = int(count)
        return dist


def load_dataset_colab(
    dataset_path,
    processed_dir="processed_homo",
    split_name="2way_base_split_target",
    fraction=0.05,
    batch_size=16,
    device="cpu"
):
    """Load dataset from Colab with specified configuration"""
    base_path = Path(dataset_path)

    # Find paths
    processed_path = base_path / processed_dir
    data_split_path = base_path / "data_split"

    raw_path = None
    for dirname in ["cpg", "raw10", "raw"]:
        candidate = base_path / dirname
        if candidate.exists():
            raw_path = candidate
            break

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed path not found: {processed_path}")
    if not data_split_path.exists():
        raise FileNotFoundError(f"Data split path not found: {data_split_path}")

    # Load split
    split_file = data_split_path / f"{split_name}.json"
    if not split_file.exists():
        split_file = data_split_path / "2way_base_split_target.json"

    with open(split_file, 'r') as f:
        split_data = json.load(f)

    label_map = split_data['label_map']
    id_map = split_data['id_map']
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']

    # Load labels
    labels = {}
    jsonl_file = raw_path / "cpg_dataset_raw10.jsonl"
    if not jsonl_file.exists():
        jsonl_file = raw_path / "cpg_dataset.jsonl"

    with open(jsonl_file, 'r') as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            target = record.get('target', 'unknown')
            labels[idx] = label_map.get(target, -1)

    # Create stratified subset
    subset_size = max(1, int(len(train_indices) * fraction))
    np.random.seed(42)
    groups = {}
    for idx in train_indices:
        label = labels.get(idx, -1)
        if label not in groups:
            groups[label] = []
        groups[label].append(idx)

    subset = []
    for label, label_indices in groups.items():
        n = max(1, int(subset_size * len(label_indices) / len(train_indices)))
        subset.extend(np.random.choice(label_indices, min(n, len(label_indices)), replace=False))

    subset = subset[:subset_size]

    # Create dataset
    tensor_path = str(processed_path / "processed.pt")
    dataset = CPGDatasetColab(tensor_path, subset, labels, id_map, device=device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    metadata = {
        'total_train': len(train_indices),
        'total_test': len(test_indices),
        'subset_size': len(subset),
        'fraction': fraction,
        'batch_size': batch_size,
        'batches': len(loader),
        'label_distribution': dataset.label_distribution,
        'device': device,
    }

    return loader, metadata, dataset


# ============================================================================
# RUN
# ============================================================================

device = DeviceManager.validate(DEVICE)

loader, metadata, dataset = load_dataset_colab(
    dataset_path=DATASET_PATH,
    processed_dir=PROCESSED_DIR,
    split_name=SPLIT_NAME,
    fraction=FRACTION,
    batch_size=BATCH_SIZE,
    device=device
)

batch = next(iter(loader))
print(f"✓ Ready! Loaded {metadata['subset_size']:,} samples on {device.upper()}")
