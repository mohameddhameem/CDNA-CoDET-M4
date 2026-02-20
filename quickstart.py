"""
QUICKSTART: Load and validate dataset with minimal memory
Run this to immediately load a small fraction for testing
"""

from load_dataset_optimized import CPGDatasetManagerOptimized
from torch.utils.data import DataLoader
import torch

def main():
    print("\n" + "█"*80)
    print("█ QUICKSTART: OPTIMIZED DATASET LOADING")
    print("█"*80)

    # 1. Initialize manager
    print("\n[1/5] Initializing dataset manager...")
    manager = CPGDatasetManagerOptimized()

    # 2. Load metadata only (< 1 second, < 1 MB)
    print("\n[2/5] Loading split data...")
    manager.load_split_data()

    # 3. Load labels (lightweight, ~10 seconds)
    print("\n[3/5] Loading labels...")
    manager.load_labels_from_jsonl()

    # 4. Validate structure
    print("\n[4/5] Validating structure...")
    is_valid = manager.validate_structure()

    if not is_valid:
        print("\n✗ Validation failed! Check the output above.")
        return

    # 5. Create small validation dataset (5%)
    print("\n[5/5] Creating validation dataset (5% of training data)...")
    val_dataset, metadata = manager.load_and_validate(fraction=0.00005)

    # Show results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"✓ Validation dataset created successfully!")
    print(f"  Total training samples: {metadata['total_train']:,}")
    print(f"  Validation samples (5%): {metadata['validation_size']:,}")
    print(f"  Label distribution: {metadata['label_distribution']}")

    # Create DataLoader
    print("\n" + "="*80)
    print("CREATING DATALOADER")
    print("="*80)
    loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"✓ DataLoader created: {len(loader)} batches of 16 samples")

    # Sample one batch (this triggers actual tensor loading)
    print("\n" + "="*80)
    print("LOADING FIRST BATCH (triggers graph tensor loading)...")
    print("="*80)

    batch = next(iter(loader))
    print(f"✓ First batch loaded successfully!")
    print(f"  Batch contents: {len(batch)} items")
    if isinstance(batch, (list, tuple)):
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                print(f"    Item {i}: Tensor {item.shape}")
            elif isinstance(item, list):
                print(f"    Item {i}: List of {len(item)} items")
            else:
                print(f"    Item {i}: {type(item).__name__}")

    print("\n" + "="*80)
    print("✓ SUCCESS! Dataset is ready for model training")
    print("="*80)
    print("""
Next steps:

1. Build your model:
   from your_model import YourModel
   model = YourModel()

2. Train on validation data:
   import torch.nn.functional as F
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   for epoch in range(10):
       for batch in loader:
           graph, label, idx, name = batch
           output = model(graph)
           loss = F.cross_entropy(output, label)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()

3. When ready for full training:
   Change fraction to 0.5 or 1.0 in:
   manager.load_and_validate(fraction=1.0)
""")


if __name__ == "__main__":
    main()
