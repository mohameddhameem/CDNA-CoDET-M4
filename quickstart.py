"""QUICKSTART: Load and validate dataset with minimal verbosity"""

from load_dataset_optimized import CPGDatasetManagerOptimized
from torch.utils.data import DataLoader
import torch

def main():
    print("Initializing dataset manager...")
    manager = CPGDatasetManagerOptimized()

    print("Loading dataset...")
    val_dataset, metadata = manager.load_and_validate(fraction=0.05)

    print(f"✓ Loaded {metadata['validation_size']:,} samples")
    print(f"  Classes: {metadata['label_distribution']}")

    loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"  DataLoader: {len(loader)} batches")

    print("\nLoading first batch...")
    batch = next(iter(loader))
    print(f"✓ Batch ready: {len(batch)} items")
    print("\n✓ Ready for training!")


if __name__ == "__main__":
    main()
