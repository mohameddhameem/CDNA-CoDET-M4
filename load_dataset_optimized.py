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
        load_mode: str = "lazy"  # 'lazy' or 'immediate'
    ):
        """
        Args:
            tensor_path: Path to processed.pt tensor file
            indices: Indices to use from the dataset
            labels_dict: Dictionary mapping index to label
            id_map: Label index to name mapping
            load_mode: 'lazy' (load on demand) or 'immediate' (load all)
        """
        self.tensor_path = tensor_path
        self.indices = indices
        self.labels_dict = labels_dict
        self.id_map = id_map
        self.load_mode = load_mode
        self._graphs = None  # Will be loaded on demand

    def _load_graphs(self):
        """Load graphs only once, on first access"""
        if self._graphs is None:
            print(f"Loading graphs from {self.tensor_path}...")
            # Use map_location='cpu' for CPU-only loading
            self._graphs = torch.load(self.tensor_path, map_location='cpu')
            print(f"  Graphs loaded successfully")

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

    def __init__(self, base_path: str = r"C:\Learning\SMU\City-of-Agents-1"):
        """Initialize the dataset manager."""
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / "dataset"
        self.processed_homo_path = self.dataset_path / "processed_homo"
        self.data_split_path = self.dataset_path / "data_split"
        self.raw_path = self.dataset_path / "cpg"

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

        print(f"✓ Base path: {self.base_path}")

    def load_split_data(self, split_name: str = "2way_base_split_target") -> Dict:
        """Load JSON split (very lightweight operation)"""
        print("\n" + "="*70)
        print("LOADING SPLIT DATA")
        print("="*70)

        split_path = self.data_split_path / f"{split_name}.json"
        print(f"Loading: {split_path}")

        with open(split_path, 'r') as f:
            self.split_data = json.load(f)

        self.label_map = self.split_data['label_map']
        self.id_map = self.split_data['id_map']

        print(f"  Label map: {self.label_map}")
        print(f"  Train: {len(self.split_data['train_indices']):,} samples")
        print(f"  Test: {len(self.split_data['test_indices']):,} samples")

        return self.split_data

    def load_labels_from_jsonl(self) -> Dict[int, int]:
        """Load labels from JSONL (lightweight if we only need specific indices)"""
        print("\n" + "="*70)
        print("LOADING LABELS")
        print("="*70)

        jsonl_path = self.raw_path / "cpg_dataset_raw10.jsonl"

        if not jsonl_path.exists():
            print(f"Warning: JSONL not found at {jsonl_path}")
            return {}

        print(f"Loading: {jsonl_path}")

        self.labels = {}
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                record = json.loads(line)
                target = record.get('target', 'unknown')
                label_idx = self.label_map.get(target, -1)
                self.labels[idx] = label_idx

        print(f"  Loaded {len(self.labels)} labels")
        return self.labels

    def create_subset_indices(
        self,
        fraction: float = 0.1,
        split: str = "train",
        stratified: bool = True
    ) -> List[int]:
        """
        Create a subset of indices from the split.

        Args:
            fraction: Fraction of data to use (0.0-1.0)
            split: 'train', 'test', or 'all'
            stratified: If True, maintains class balance in subset

        Returns:
            List of indices for the subset
        """
        print("\n" + "="*70)
        print("CREATING SUBSET")
        print("="*70)

        if self.split_data is None:
            raise RuntimeError("Must load split data first")

        # Get indices
        if split == "train":
            all_indices = self.split_data['train_indices']
        elif split == "test":
            all_indices = self.split_data['test_indices']
        else:  # 'all'
            all_indices = (
                self.split_data['train_indices'] +
                self.split_data['test_indices']
            )

        subset_size = max(1, int(len(all_indices) * fraction))

        if stratified and self.labels:
            # Maintain class balance
            subset = self._stratified_sample(all_indices, subset_size)
        else:
            # Random sample
            np.random.seed(42)
            subset = list(np.random.choice(all_indices, subset_size, replace=False))

        print(f"  Original size: {len(all_indices):,}")
        print(f"  Subset size: {len(subset):,} ({fraction*100:.1f}%)")

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
        """
        Create a validation dataset with only a fraction of data.

        Args:
            fraction: Fraction of data (0.1 = 10%)
            split: 'train' or 'test'

        Returns:
            CPGDatasetOptimized instance
        """
        if self.split_data is None or self.labels is None:
            raise RuntimeError("Must load split data and labels first")

        # Create subset indices
        subset_indices = self.create_subset_indices(
            fraction=fraction,
            split=split,
            stratified=True
        )

        # Create dataset with lazy loading
        tensor_path = str(self.processed_homo_path / "processed.pt")
        dataset = CPGDatasetOptimized(
            tensor_path=tensor_path,
            indices=subset_indices,
            labels_dict=self.labels,
            id_map=self.id_map,
            load_mode="lazy"
        )

        print(f"\n✓ Validation dataset created with {len(dataset)} samples")
        print(f"  Label distribution: {dataset.label_distribution}")

        return dataset

    def validate_structure(self) -> bool:
        """Quick validation without loading actual data"""
        print("\n" + "="*70)
        print("VALIDATING DATASET STRUCTURE")
        print("="*70)

        checks = []

        # Check tensor file exists and has valid header
        tensor_path = self.processed_homo_path / "processed.pt"
        if tensor_path.exists():
            size_gb = tensor_path.stat().st_size / (1024**3)
            print(f"✓ Tensor file exists: {size_gb:.2f} GB")
            checks.append(True)
        else:
            print(f"✗ Tensor file missing: {tensor_path}")
            checks.append(False)

        # Check split data validity
        if self.split_data:
            train_size = len(self.split_data['train_indices'])
            test_size = len(self.split_data['test_indices'])
            total = train_size + test_size

            print(f"✓ Split valid: {train_size:,} train + {test_size:,} test = {total:,} total")
            checks.append(True)
        else:
            print(f"✗ Split data not loaded")
            checks.append(False)

        # Check labels loaded
        if self.labels:
            print(f"✓ Labels loaded: {len(self.labels)} labels")
            checks.append(True)
        else:
            print(f"✗ Labels not loaded")
            checks.append(False)

        # Check no overlap
        if self.split_data and self.labels:
            train_set = set(self.split_data['train_indices'])
            test_set = set(self.split_data['test_indices'])
            overlap = train_set & test_set
            if not overlap:
                print(f"✓ No train/test overlap")
                checks.append(True)
            else:
                print(f"✗ Found {len(overlap)} overlapping indices")
                checks.append(False)

        all_valid = all(checks)
        status = "✓ VALID" if all_valid else "✗ INVALID"
        print(f"\n{status}")

        return all_valid

    def load_and_validate(self, fraction: float = 0.05) -> Tuple[CPGDatasetOptimized, Dict]:
        """
        Complete pipeline: load, validate, and create validation dataset.

        Args:
            fraction: Fraction of data for validation (e.g., 0.05 = 5%)

        Returns:
            Tuple of (validation_dataset, metadata)
        """
        print("\n" + "█"*70)
        print("█ OPTIMIZED DATASET LOADING PIPELINE")
        print("█"*70)

        # Check memory before
        mem_before = self.get_memory_info()
        print(f"\nMemory before loading: {mem_before['rss_mb']:.0f} MB available")

        # Load metadata (lightweight)
        self.load_split_data()
        self.load_labels_from_jsonl()

        # Validate structure
        self.validate_structure()

        # Create small validation dataset
        val_dataset = self.create_validation_dataset(fraction=fraction, split="train")

        # Check memory after
        mem_after = self.get_memory_info()
        print(f"\nMemory after loading: {mem_after['rss_mb']:.0f} MB used")
        print(f"Memory increase: {(mem_before['rss_mb'] - mem_after['rss_mb']):.0f} MB")

        metadata = {
            'total_train': len(self.split_data['train_indices']),
            'total_test': len(self.split_data['test_indices']),
            'validation_size': len(val_dataset),
            'fraction': fraction,
            'label_distribution': val_dataset.label_distribution
        }

        print("\n" + "="*70)
        print("METADATA")
        print("="*70)
        for key, val in metadata.items():
            print(f"  {key}: {val}")

        return val_dataset, metadata


def quick_validation():
    """Quick validation without loading heavy data"""
    manager = CPGDatasetManagerOptimized()

    # Load only metadata
    manager.load_split_data()
    manager.load_labels_from_jsonl()

    # Validate
    manager.validate_structure()

    print("\n✓ Dataset is ready for use!")
    return manager


def main():
    """Example: Load only 5% of data for validation"""
    manager = CPGDatasetManagerOptimized()

    # Load and validate (only loads 5% of data)
    val_dataset, metadata = manager.load_and_validate(fraction=0.05)

    print("\n" + "="*70)
    print("CREATING VALIDATION DATALOADER")
    print("="*70)

    # Create dataloader
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Batches: {len(val_loader)}")

    # Sample one batch (this is when actual graph data is loaded)
    print("\nLoading first batch...")
    batch = next(iter(val_loader))
    print(f"  Batch size: {len(batch)} items per batch")

    # Cleanup
    gc.collect()
    print("\n✓ Done! Memory cleaned up.")


if __name__ == "__main__":
    main()
