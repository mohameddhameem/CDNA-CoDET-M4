"""
Optimized CPG Dataset Loader for CPU with Lazy Loading
This module loads only required data fractions for validation.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import psutil
import gc


class CPGDatasetOptimized(Dataset):
    """Memory-efficient PyTorch Dataset with lazy loading"""

    def __init__(
        self,
        tensor_path: str,
        indices: List[int],
        labels_dict: Dict[int, int],
        id_map: Dict[str, str],
        load_mode: str = "lazy",
        device: str = "cpu"
    ):
        """
        Args:
            tensor_path: Path to processed.pt tensor file
            indices: Indices to use from the dataset
            labels_dict: Dictionary mapping index to label
            id_map: Label index to name mapping
            load_mode: 'lazy' (load on demand) or 'immediate' (load all)
            device: 'cpu' or 'cuda'
        """
        self.tensor_path = tensor_path
        self.indices = indices
        self.labels_dict = labels_dict
        self.id_map = id_map
        self.load_mode = load_mode
        self.device = device
        self._graphs = None

    def _load_graphs(self):
        """Load graphs only once, on first access"""
        if self._graphs is None:
            self._graphs = torch.load(self.tensor_path, map_location=self.device)

    @property
    def graphs(self):
        """Lazy load graphs"""
        if self.load_mode == "lazy":
            self._load_graphs()
        return self._graphs

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple:
        """Get single sample with lazy loading"""
        self._load_graphs()  # Ensure graphs are loaded

        graph_idx = self.indices[idx]
        graph = self._graphs[graph_idx]
        label = self.labels_dict.get(graph_idx, -1)
        label_name = self.id_map.get(str(label), 'unknown')

        return graph, label, graph_idx, label_name

    @property
    def label_distribution(self) -> Dict[str, int]:
        """Get label distribution without loading graphs"""
        labels = [self.labels_dict.get(idx, -1) for idx in self.indices]
        unique, counts = np.unique(labels, return_counts=True)

        dist = {}
        for label_idx, count in zip(unique, counts):
            label_name = self.id_map.get(str(label_idx), 'unknown')
            dist[label_name] = int(count)

        return dist


class CPGDatasetManagerOptimized:
    """Optimized manager for loading only necessary data fractions"""

    def __init__(self, base_path: str = r"C:\Learning\SMU\City-of-Agents-1", device: str = "auto"):
        """Initialize the dataset manager.

        Args:
            base_path: Path to dataset root
            device: 'auto' (auto-detect), 'cuda' (GPU), or 'cpu'
        """
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / "dataset"
        self.processed_homo_path = self.dataset_path / "processed_homo"
        self.data_split_path = self.dataset_path / "data_split"
        self.raw_path = self.dataset_path / "cpg"

        # Handle device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("⚠ CUDA requested but not available. Using CPU.")
                self.device = "cpu"
            else:
                self.device = "cuda"
        else:
            self.device = "cpu"

        self._verify_paths()

        # Data containers
        self.split_data = None
        self.labels = None
        self.label_map = None
        self.id_map = None

    def _verify_paths(self):
        """Verify required paths exist"""
        if not self.processed_homo_path.exists():
            raise FileNotFoundError(f"Path not found: {self.processed_homo_path}")
        if not self.data_split_path.exists():
            raise FileNotFoundError(f"Path not found: {self.data_split_path}")

    def load_split_data(self, split_name: str = "2way_base_split_target") -> Dict:
        """Load JSON split"""
        split_path = self.data_split_path / f"{split_name}.json"

        with open(split_path, 'r') as f:
            self.split_data = json.load(f)

        self.label_map = self.split_data['label_map']
        self.id_map = self.split_data['id_map']

        return self.split_data

    def load_labels_from_jsonl(self) -> Dict[int, int]:
        """Load labels from JSONL"""
        jsonl_path = self.raw_path / "cpg_dataset_raw10.jsonl"

        if not jsonl_path.exists():
            return {}

        self.labels = {}
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                record = json.loads(line)
                target = record.get('target', 'unknown')
                label_idx = self.label_map.get(target, -1)
                self.labels[idx] = label_idx

        return self.labels

    def create_subset_indices(
        self,
        fraction: float = 0.1,
        split: str = "train",
        stratified: bool = True
    ) -> List[int]:
        """Create a subset of indices from the split"""
        if self.split_data is None:
            raise RuntimeError("Must load split data first")

        if split == "train":
            all_indices = self.split_data['train_indices']
        elif split == "test":
            all_indices = self.split_data['test_indices']
        else:
            all_indices = (
                self.split_data['train_indices'] +
                self.split_data['test_indices']
            )

        subset_size = max(1, int(len(all_indices) * fraction))

        if stratified and self.labels:
            subset = self._stratified_sample(all_indices, subset_size)
        else:
            np.random.seed(42)
            subset = list(np.random.choice(all_indices, subset_size, replace=False))

        return sorted(subset)

    def _stratified_sample(self, indices: List[int], n_samples: int) -> List[int]:
        """Sample indices while maintaining class balance"""
        # Group by label
        groups = {}
        for idx in indices:
            label = self.labels.get(idx, -1)
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)

        # Sample from each group proportionally
        subset = []
        for label, label_indices in groups.items():
            n = max(1, int(n_samples * len(label_indices) / len(indices)))
            subset.extend(np.random.choice(label_indices, n, replace=False))

        return subset[:n_samples]

    def get_memory_info(self) -> Dict:
        """Get current memory usage"""
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }

    def create_validation_dataset(
        self,
        fraction: float = 0.1,
        split: str = "train"
    ) -> CPGDatasetOptimized:
        """Create a validation dataset with only a fraction of data"""
        if self.split_data is None or self.labels is None:
            raise RuntimeError("Must load split data and labels first")

        subset_indices = self.create_subset_indices(
            fraction=fraction,
            split=split,
            stratified=True
        )

        tensor_path = str(self.processed_homo_path / "processed.pt")
        dataset = CPGDatasetOptimized(
            tensor_path=tensor_path,
            indices=subset_indices,
            labels_dict=self.labels,
            id_map=self.id_map,
            load_mode="lazy",
            device=self.device
        )

        return dataset

    def validate_structure(self) -> bool:
        """Quick validation without loading actual data"""
        checks = []

        tensor_path = self.processed_homo_path / "processed.pt"
        if tensor_path.exists():
            checks.append(True)
        else:
            checks.append(False)

        if self.split_data:
            checks.append(True)
        else:
            checks.append(False)

        if self.labels:
            checks.append(True)
        else:
            checks.append(False)

        if self.split_data and self.labels:
            train_set = set(self.split_data['train_indices'])
            test_set = set(self.split_data['test_indices'])
            overlap = train_set & test_set
            checks.append(len(overlap) == 0)

        return all(checks)

    def load_and_validate(self, fraction: float = 0.05) -> Tuple[CPGDatasetOptimized, Dict]:
        """Complete pipeline: load, validate, and create validation dataset"""
        self.load_split_data()
        self.load_labels_from_jsonl()
        self.validate_structure()

        val_dataset = self.create_validation_dataset(fraction=fraction, split="train")

        metadata = {
            'total_train': len(self.split_data['train_indices']),
            'total_test': len(self.split_data['test_indices']),
            'validation_size': len(val_dataset),
            'fraction': fraction,
            'label_distribution': val_dataset.label_distribution
        }

        return val_dataset, metadata


def quick_validation():
    """Quick validation without loading heavy data"""
    manager = CPGDatasetManagerOptimized()
    manager.load_split_data()
    manager.load_labels_from_jsonl()
    manager.validate_structure()
    return manager


def main():
    """Example: Load only 5% of data for validation"""
    manager = CPGDatasetManagerOptimized(device="auto")
    val_dataset, metadata = manager.load_and_validate(fraction=0.05)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"✓ Loaded {metadata['validation_size']:,} samples on {manager.device.upper()}")


if __name__ == "__main__":
    main()
