"""
Multi-Environment Dataset Loader (Colab + Local)
Automatically detects environment and supports custom paths
GPU/CPU configurable
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


class EnvironmentDetector:
    """Detect and configure environment (Colab vs Local)"""

    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    @staticmethod
    def get_default_path() -> Path:
        """Get default dataset path based on environment"""
        if EnvironmentDetector.is_colab():
            return Path("/content/drive/MyDrive/CPG")
        else:
            return Path(r"C:\Learning\SMU\City-of-Agents-1\dataset")

    @staticmethod
    def get_default_device() -> str:
        """Get default device based on environment"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


class CPGDatasetOptimizedConfigurable(Dataset):
    """Configurable memory-efficient dataset with lazy loading"""

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
            tensor_path: Path to processed.pt
            indices: Indices to use
            labels_dict: Index to label mapping
            id_map: Label index to name
            load_mode: 'lazy' or 'immediate'
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
        """Load graphs on first access"""
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
        self._load_graphs()
        graph_idx = self.indices[idx]
        graph = self._graphs[graph_idx]
        label = self.labels_dict.get(graph_idx, -1)
        label_name = self.id_map.get(str(label), 'unknown')
        return graph, label, graph_idx, label_name

    @property
    def label_distribution(self) -> Dict[str, int]:
        """Get label distribution"""
        labels = [self.labels_dict.get(idx, -1) for idx in self.indices]
        unique, counts = np.unique(labels, return_counts=True)

        dist = {}
        for label_idx, count in zip(unique, counts):
            label_name = self.id_map.get(str(label_idx), 'unknown')
            dist[label_name] = int(count)

        return dist


class CPGDatasetManagerConfigurable:
    """Configurable manager supporting Colab and local paths"""

    def __init__(
        self,
        base_path: Optional[str] = None,
        processed_dir: str = "processed_homo",
        split_name: str = "2way_base_split_target",
        raw_jsonl: str = "cpg_dataset_raw10.jsonl",
        verbose: bool = False,
        device: str = "auto"
    ):
        """
        Initialize dataset manager with flexible configuration.

        Args:
            base_path: Custom path to dataset root. If None, uses auto-detected default.
            processed_dir: Name of processed directory
            split_name: Name of JSON split file (without .json)
            raw_jsonl: Name of raw JSONL file
            verbose: Print detailed information
            device: 'auto' (auto-detect), 'cuda' (GPU), or 'cpu'
        """
        self.verbose = verbose
        self.split_name = split_name
        self.raw_jsonl = raw_jsonl

        # Validate device
        if device == "auto":
            self.device = EnvironmentDetector.get_default_device()
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("⚠ CUDA requested but not available. Using CPU.")
                self.device = "cpu"
            else:
                self.device = "cuda"
        else:
            self.device = "cpu"

        # Set base path
        if base_path is None:
            base_path = str(EnvironmentDetector.get_default_path().parent)

        self.base_path = Path(base_path)

        # Build paths
        self.processed_dir = processed_dir
        self.processed_path = self.base_path / processed_dir
        self.data_split_path = self.base_path / "data_split"
        self.raw_path = self._find_raw_directory()

        # Data containers
        self.split_data = None
        self.labels = None
        self.label_map = None
        self.id_map = None

    def _find_raw_directory(self) -> Path:
        """Find raw data directory (could be 'cpg', 'raw', or 'raw10')"""
        possible_dirs = ["cpg", "raw10", "raw"]
        for dirname in possible_dirs:
            candidate = self.base_path / dirname
            if candidate.exists():
                return candidate
        return self.base_path / "cpg"

    def load_split_data(self) -> Dict:
        """Load JSON split file"""
        # Try different naming conventions
        possible_paths = [
            self.data_split_path / f"{self.split_name}.json",
            self.data_split_path / "2way_base_split_target.json",
            self.data_split_path / "2way_base_split_model.json",
        ]

        split_path = None
        for p in possible_paths:
            if p.exists():
                split_path = p
                break

        if split_path is None:
            raise FileNotFoundError(
                f"Could not find split JSON in {self.data_split_path}"
            )

        with open(split_path, 'r') as f:
            self.split_data = json.load(f)

        self.label_map = self.split_data['label_map']
        self.id_map = self.split_data['id_map']

        if self.verbose:
            print(f"Loaded split: {split_path.name}")
            print(f"  Train: {len(self.split_data['train_indices']):,} | Test: {len(self.split_data['test_indices']):,}")

        return self.split_data

    def load_labels_from_jsonl(self) -> Dict[int, int]:
        """Load labels from JSONL file"""
        # Try different paths
        possible_paths = [
            self.raw_path / self.raw_jsonl,
            self.raw_path / "cpg_dataset_raw10.jsonl",
            self.raw_path / "cpg_dataset.jsonl",
            Path(self.base_path.parent) / self.raw_jsonl,
        ]

        jsonl_path = None
        for p in possible_paths:
            if p.exists():
                jsonl_path = p
                break

        if jsonl_path is None:
            print(f"⚠ Warning: JSONL file not found")
            return {}

        self.labels = {}
        line_count = 0

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    record = json.loads(line)
                    target = record.get('target', 'unknown')
                    label_idx = self.label_map.get(target, -1)
                    self.labels[idx] = label_idx
                    line_count = idx + 1

            if self.verbose:
                print(f"Loaded labels: {len(self.labels):,}")

        except Exception as e:
            print(f"Error loading JSONL: {e}")
            return {}

        return self.labels

    def create_subset_indices(
        self,
        fraction: float = 0.1,
        split: str = "train",
        stratified: bool = True,
        seed: int = 42
    ) -> List[int]:
        """Create subset of indices"""
        if self.split_data is None:
            raise RuntimeError("Must load split data first")

        np.random.seed(seed)

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
            subset = list(np.random.choice(all_indices, subset_size, replace=False))

        return sorted(subset)

    def _stratified_sample(self, indices: List[int], n_samples: int) -> List[int]:
        """Stratified sampling to maintain class balance"""
        groups = {}
        for idx in indices:
            label = self.labels.get(idx, -1)
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)

        subset = []
        for label, label_indices in groups.items():
            n = max(1, int(n_samples * len(label_indices) / len(indices)))
            sampled = np.random.choice(label_indices, min(n, len(label_indices)), replace=False)
            subset.extend(sampled)

        return subset[:n_samples]

    def create_dataset(
        self,
        indices: List[int],
        load_mode: str = "lazy"
    ) -> CPGDatasetOptimizedConfigurable:
        """Create dataset from indices"""
        if self.labels is None:
            raise RuntimeError("Must load labels first")

        tensor_path = str(self.processed_path / "processed.pt")

        dataset = CPGDatasetOptimizedConfigurable(
            tensor_path=tensor_path,
            indices=indices,
            labels_dict=self.labels,
            id_map=self.id_map,
            load_mode=load_mode,
            device=self.device
        )

        return dataset

    def load_and_validate(
        self,
        fraction: float = 0.05,
        split: str = "train"
    ) -> Tuple[CPGDatasetOptimizedConfigurable, Dict]:
        """Complete load and validate pipeline"""
        # Load metadata
        self.load_split_data()
        self.load_labels_from_jsonl()

        # Create subset
        indices = self.create_subset_indices(fraction=fraction, split=split)

        # Create dataset
        dataset = self.create_dataset(indices)

        metadata = {
            'total_train': len(self.split_data['train_indices']),
            'total_test': len(self.split_data['test_indices']),
            'validation_size': len(dataset),
            'fraction': fraction,
            'split': split,
            'label_distribution': dataset.label_distribution
        }

        return dataset, metadata


def main():
    """Example: Run with default auto-detected environment"""
    manager = CPGDatasetManagerConfigurable()
    dataset, metadata = manager.load_and_validate(fraction=0.05, split="train")
    print(f"✓ Loaded {metadata['validation_size']:,} samples on {manager.device.upper()}")


if __name__ == "__main__":
    main()
