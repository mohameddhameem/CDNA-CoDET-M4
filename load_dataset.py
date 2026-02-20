"""
Local CPG Dataset Loader
This module loads and processes the CPG dataset from local tensors and JSON splits.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


class CPGDatasetLocal(Dataset):
    """PyTorch Dataset for CPG (Code Property Graphs) with labels from JSON splits"""

    def __init__(
        self,
        graphs: List,
        indices: List[int],
        labels_dict: Dict[int, int],
        id_map: Dict[str, str]
    ):
        """
        Args:
            graphs: List of graph objects/tensors from processed.pt
            indices: List of indices to use from the dataset
            labels_dict: Dictionary mapping graph index to label
            id_map: Dictionary mapping label index to label name (e.g., {'0': 'ai', '1': 'human'})
        """
        self.graphs = graphs
        self.indices = indices
        self.labels_dict = labels_dict
        self.id_map = id_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            graph: Graph object/tensor
            label: Integer label (0 or 1)
            graph_idx: Original index in the full dataset
            label_name: String label name ('ai' or 'human')
        """
        graph_idx = self.indices[idx]
        graph = self.graphs[graph_idx]
        label = self.labels_dict.get(graph_idx, -1)
        label_name = self.id_map.get(str(label), 'unknown')

        return graph, label, graph_idx, label_name

    @property
    def label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in this split"""
        labels = [self.labels_dict.get(idx, -1) for idx in self.indices]
        unique, counts = np.unique(labels, return_counts=True)

        dist = {}
        for label_idx, count in zip(unique, counts):
            label_name = self.id_map.get(str(label_idx), 'unknown')
            dist[label_name] = int(count)

        return dist


class CPGDatasetManager:
    """Manager for loading and handling CPG dataset locally"""

    def __init__(self, base_path: str = r"C:\Learning\SMU\City-of-Agents-1"):
        """
        Initialize the dataset manager.

        Args:
            base_path: Path to the repository root
        """
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / "dataset"
        self.processed_homo_path = self.dataset_path / "processed_homo"
        self.data_split_path = self.dataset_path / "data_split"
        self.raw_path = self.dataset_path / "cpg"

        # Verify paths exist
        self._verify_paths()

        # Load data
        self.graphs = None
        self.split_data = None
        self.labels = None
        self.label_map = None
        self.id_map = None

    def _verify_paths(self):
        """Verify that required paths exist"""
        required_paths = [
            self.processed_homo_path,
            self.data_split_path,
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")

        print(f"✓ Base path verified: {self.base_path}")
        print(f"✓ Dataset path verified: {self.dataset_path}")

    def load_tensor_data(self) -> Dict:
        """
        Load tensor files from processed_homo directory.

        Returns:
            Dictionary with keys: processed, pre_filter, pre_transform
        """
        print("\n" + "="*60)
        print("LOADING TENSOR DATA")
        print("="*60)

        tensor_data = {}

        # Load processed.pt
        processed_path = self.processed_homo_path / "processed.pt"
        print(f"\nLoading: {processed_path}")
        self.graphs = torch.load(processed_path, map_location='cpu')
        tensor_data['processed'] = self.graphs

        if isinstance(self.graphs, list):
            print(f"  Type: List with {len(self.graphs)} graphs")
        elif isinstance(self.graphs, dict):
            print(f"  Type: Dictionary with keys: {list(self.graphs.keys())}")
        else:
            print(f"  Type: {type(self.graphs)}")
            if hasattr(self.graphs, 'shape'):
                print(f"  Shape: {self.graphs.shape}")

        # Load pre_filter.pt
        pre_filter_path = self.processed_homo_path / "pre_filter.pt"
        if pre_filter_path.exists():
            print(f"\nLoading: {pre_filter_path}")
            tensor_data['pre_filter'] = torch.load(pre_filter_path, map_location='cpu')
            print(f"  Type: {type(tensor_data['pre_filter'])}")

        # Load pre_transform.pt
        pre_transform_path = self.processed_homo_path / "pre_transform.pt"
        if pre_transform_path.exists():
            print(f"\nLoading: {pre_transform_path}")
            tensor_data['pre_transform'] = torch.load(pre_transform_path, map_location='cpu')
            print(f"  Type: {type(tensor_data['pre_transform'])}")

        return tensor_data

    def load_split_data(self, split_name: str = "2way_base_split_target") -> Dict:
        """
        Load JSON split file.

        Args:
            split_name: Name of the split file (without .json extension)

        Returns:
            Dictionary with split data
        """
        print("\n" + "="*60)
        print("LOADING SPLIT DATA")
        print("="*60)

        split_path = self.data_split_path / f"{split_name}.json"
        print(f"\nLoading: {split_path}")

        with open(split_path, 'r') as f:
            self.split_data = json.load(f)

        self.label_map = self.split_data['label_map']
        self.id_map = self.split_data['id_map']

        print(f"  Label Map: {self.label_map}")
        print(f"  Train indices: {len(self.split_data['train_indices'])}")
        print(f"  Test indices: {len(self.split_data['test_indices'])}")
        print(f"  Total: {len(self.split_data['train_indices']) + len(self.split_data['test_indices'])}")

        return self.split_data

    def load_labels_from_jsonl(self) -> Dict[int, int]:
        """
        Load labels from the original JSONL file.

        Returns:
            Dictionary mapping graph index to label integer
        """
        print("\n" + "="*60)
        print("LOADING LABELS FROM JSONL")
        print("="*60)

        jsonl_path = self.raw_path / "cpg_dataset_raw10.jsonl"

        if not jsonl_path.exists():
            print(f"Warning: JSONL file not found at {jsonl_path}")
            return {}

        print(f"\nLoading: {jsonl_path}")

        self.labels = {}
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                record = json.loads(line)
                target = record.get('target', 'unknown')
                label_idx = self.label_map.get(target, -1)
                self.labels[idx] = label_idx

        print(f"  Loaded {len(self.labels)} labels")

        return self.labels

    def create_datasets(self) -> Tuple[CPGDatasetLocal, CPGDatasetLocal]:
        """
        Create train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if self.graphs is None or self.split_data is None or self.labels is None:
            raise RuntimeError("Must load tensor data, split data, and labels first")

        print("\n" + "="*60)
        print("CREATING DATASETS")
        print("="*60)

        train_indices = self.split_data['train_indices']
        test_indices = self.split_data['test_indices']

        train_dataset = CPGDatasetLocal(
            self.graphs,
            train_indices,
            self.labels,
            self.id_map
        )

        test_dataset = CPGDatasetLocal(
            self.graphs,
            test_indices,
            self.labels,
            self.id_map
        )

        print(f"\nTrain Dataset:")
        print(f"  Size: {len(train_dataset)}")
        print(f"  Label Distribution: {train_dataset.label_distribution}")

        print(f"\nTest Dataset:")
        print(f"  Size: {len(test_dataset)}")
        print(f"  Label Distribution: {test_dataset.label_distribution}")

        return train_dataset, test_dataset

    def analyze_sample(self, dataset: CPGDatasetLocal, sample_idx: int = 0):
        """
        Analyze a single sample from the dataset.

        Args:
            dataset: Dataset to sample from
            sample_idx: Index within the dataset
        """
        print("\n" + "="*60)
        print("ANALYZING SAMPLE")
        print("="*60)

        graph, label, graph_idx, label_name = dataset[sample_idx]

        print(f"\nDataset Index: {sample_idx}")
        print(f"Graph Index (in full dataset): {graph_idx}")
        print(f"Label: {label} ({label_name})")
        print(f"Graph Type: {type(graph)}")

        # If it's a torch geometric Data object
        if hasattr(graph, 'keys'):
            print(f"\nGraph Attributes:")
            for key in graph.keys():
                attr = graph[key]
                if isinstance(attr, torch.Tensor):
                    print(f"  {key}: Tensor {attr.shape}, dtype={attr.dtype}")
                else:
                    print(f"  {key}: {type(attr)}")
        elif isinstance(graph, torch.Tensor):
            print(f"  Shape: {graph.shape}")
            print(f"  Dtype: {graph.dtype}")
        else:
            # Try to print attributes
            print(f"\nGraph Attributes:")
            for attr_name in dir(graph):
                if not attr_name.startswith('_'):
                    try:
                        val = getattr(graph, attr_name)
                        if isinstance(val, torch.Tensor):
                            print(f"  {attr_name}: Tensor {val.shape}, dtype={val.dtype}")
                        elif isinstance(val, (int, str, float, bool)):
                            print(f"  {attr_name}: {val}")
                    except:
                        pass

    def load_all(self) -> Tuple[CPGDatasetLocal, CPGDatasetLocal]:
        """
        Load everything and return train/test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        self.load_tensor_data()
        self.load_split_data()
        self.load_labels_from_jsonl()
        train_dataset, test_dataset = self.create_datasets()

        return train_dataset, test_dataset


def main():
    """Example usage of the dataset manager"""

    # Initialize manager
    manager = CPGDatasetManager()

    # Load all data
    train_dataset, test_dataset = manager.load_all()

    # Analyze first sample
    manager.analyze_sample(train_dataset, sample_idx=0)

    # Create DataLoaders (if needed)
    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"\nTrain DataLoader: {len(train_loader)} batches (batch_size=32)")
    print(f"Test DataLoader: {len(test_loader)} batches (batch_size=32)")

    # Sample a batch
    print("\n" + "="*60)
    print("SAMPLING A BATCH")
    print("="*60)

    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    if isinstance(batch, (list, tuple)):
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                print(f"  Item {i}: Tensor of shape {item.shape}")
            elif isinstance(item, list):
                print(f"  Item {i}: List with {len(item)} elements")
            else:
                print(f"  Item {i}: {type(item)}")


if __name__ == "__main__":
    main()
