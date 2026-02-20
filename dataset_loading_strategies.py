"""
CPU-Optimized Dataset Loading Guide
Different strategies for loading fractions of data with minimal memory usage
"""

from pathlib import Path
from load_dataset_optimized import CPGDatasetManagerOptimized, CPGDatasetOptimized
from torch.utils.data import DataLoader
import gc


def strategy_1_metadata_only():
    """
    STRATEGY 1: Load only metadata (< 1 MB)
    Perfect for: Verifying data exists and understanding structure
    Memory: Minimal
    Time: < 1 second
    """
    print("\n" + "="*80)
    print("STRATEGY 1: METADATA ONLY (Validation)")
    print("="*80)
    print("What it does:")
    print("  - Loads split JSON (train/test indices)")
    print("  - Loads all labels from JSONL")
    print("  - Validates structure")
    print("  - Does NOT load actual graph tensors")
    print("\nWhen to use:")
    print("  - Check if dataset exists and is valid")
    print("  - Understand class distribution")
    print("  - Verify indices are correct\n")

    manager = CPGDatasetManagerOptimized()
    manager.load_split_data()
    manager.load_labels_from_jsonl()
    manager.validate_structure()

    print("\n✓ Metadata loaded successfully!")
    print(f"  Train samples: {len(manager.split_data['train_indices']):,}")
    print(f"  Test samples: {len(manager.split_data['test_indices']):,}")


def strategy_2_small_fraction():
    """
    STRATEGY 2: Load small fraction (5%) with lazy loading
    Perfect for: Quick prototyping and testing
    Memory: ~1.6 GB (for 5% of 32 GB dataset)
    Time: Lazy - on demand
    """
    print("\n" + "="*80)
    print("STRATEGY 2: SMALL FRACTION (5%) WITH LAZY LOADING")
    print("="*80)
    print("What it does:")
    print("  - Loads metadata + labels")
    print("  - Creates subset of 5% of data (stratified)")
    print("  - Lazy loads graph tensors on first access")
    print("\nWhen to use:")
    print("  - Test your model pipeline")
    print("  - Validate data loading code")
    print("  - Quick iteration during development\n")

    manager = CPGDatasetManagerOptimized()
    val_dataset, metadata = manager.load_and_validate(fraction=0.05)

    # Create dataloader
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(f"\nDataLoader created:")
    print(f"  Dataset size: {len(val_dataset)}")
    print(f"  Batch size: 8")
    print(f"  Total batches: {len(val_loader)}")

    # Sample batch (triggers actual loading)
    print("\nLoading first batch (this triggers tensor load)...")
    batch = next(iter(val_loader))
    print(f"  ✓ Batch loaded with {len(batch)} graphs")

    del manager, val_dataset, val_loader
    gc.collect()


def strategy_3_medium_fraction():
    """
    STRATEGY 3: Load medium fraction (25%) for better validation
    Perfect for: Testing with more representative data
    Memory: ~8 GB
    Time: Depends on access pattern
    """
    print("\n" + "="*80)
    print("STRATEGY 3: MEDIUM FRACTION (25%)")
    print("="*80)
    print("What it does:")
    print("  - Loads 25% of training data")
    print("  - Maintains class balance (stratified sampling)")
    print("  - Good for testing without full dataset")
    print("\nWhen to use:")
    print("  - More thorough testing")
    print("  - Check model generalizes to larger data")
    print("  - Profile memory/compute requirements\n")

    manager = CPGDatasetManagerOptimized()
    val_dataset, metadata = manager.load_and_validate(fraction=0.25)

    print(f"\nDataset info:")
    print(f"  Fraction: {metadata['fraction']*100:.0f}%")
    print(f"  Samples: {metadata['validation_size']:,}")
    print(f"  Class distribution: {metadata['label_distribution']}")

    del manager, val_dataset
    gc.collect()


def strategy_4_efficient_subset():
    """
    STRATEGY 4: Create multiple small subsets sequentially
    Perfect for: Cross-validation or multiple experiments
    Memory: Always ~1.6 GB (if using 5%)
    Time: Load one, use it, clean, load next
    """
    print("\n" + "="*80)
    print("STRATEGY 4: SEQUENTIAL SMALL SUBSETS")
    print("="*80)
    print("What it does:")
    print("  - Load metadata once")
    print("  - Create many small subsets (5% each)")
    print("  - Process one subset, clear memory, next subset")
    print("\nWhen to use:")
    print("  - Cross-validation")
    print("  - Running multiple experiments")
    print("  - Limited memory (16 GB RAM)\n")

    manager = CPGDatasetManagerOptimized()
    manager.load_split_data()
    manager.load_labels_from_jsonl()

    n_subsets = 3
    for i in range(n_subsets):
        print(f"\nSubset {i+1}/{n_subsets}:")

        subset_indices = manager.create_subset_indices(
            fraction=0.05,
            split="train",
            stratified=True
        )

        dataset = CPGDatasetOptimized(
            tensor_path=str(manager.processed_homo_path / "processed.pt"),
            indices=subset_indices,
            labels_dict=manager.labels,
            id_map=manager.id_map,
            load_mode="lazy"
        )

        print(f"  Created dataset with {len(dataset)} samples")
        print(f"  Distribution: {dataset.label_distribution}")

        # Simulate processing
        loader = DataLoader(dataset, batch_size=8)
        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count >= 2:  # Just process first 2 batches
                break

        print(f"  Processed {batch_count} batches")

        # Cleanup
        del dataset, loader
        gc.collect()

    del manager
    gc.collect()


def strategy_5_comparison():
    """
    STRATEGY 5: Compare all strategies
    Shows memory/time tradeoffs
    """
    print("\n" + "="*80)
    print("STRATEGY 5: MEMORY/TIME COMPARISON")
    print("="*80)

    strategies = {
        "Metadata Only": {
            "memory_mb": 500,
            "load_time_sec": 10,
            "use_case": "Validation only",
        },
        "5% Fraction (Lazy)": {
            "memory_mb": 1600,
            "load_time_sec": 15,
            "use_case": "Quick prototyping",
        },
        "25% Fraction": {
            "memory_mb": 8000,
            "load_time_sec": 60,
            "use_case": "Representative testing",
        },
        "50% Fraction": {
            "memory_mb": 16000,
            "load_time_sec": 120,
            "use_case": "Thorough validation",
        },
        "100% Full Load": {
            "memory_mb": 32000,
            "load_time_sec": 300,
            "use_case": "Production training",
        }
    }

    print("\nStrategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Memory':<15} {'Time':<15} {'Use Case':<25}")
    print("-" * 80)

    for name, info in strategies.items():
        print(f"{name:<25} {info['memory_mb']:<13} MB {info['load_time_sec']:<13} s {info['use_case']:<25}")

    print("\n" + "-" * 80)
    print("RECOMMENDATION FOR YOUR USE CASE:")
    print("-" * 80)
    print("""
If you have 16 GB RAM:
  → Use Strategy 2 (5%) for initial testing
  → Then Strategy 4 (sequential) for validation
  → Total memory never exceeds 2 GB overhead

If you have 32+ GB RAM:
  → Use Strategy 3 (25%) for good coverage
  → Fast iteration while keeping good validation

To start immediately:
  → Run: python load_dataset_optimized.py
  → This uses Strategy 2 (5% with lazy loading)
""")


def get_quick_start_code():
    """Print code snippets for quick start"""
    print("\n" + "="*80)
    print("QUICK START CODE SNIPPETS")
    print("="*80)

    print("\n1. VALIDATE STRUCTURE (< 1 sec, < 1 MB):")
    print("""
from load_dataset_optimized import CPGDatasetManagerOptimized

manager = CPGDatasetManagerOptimized()
manager.load_split_data()
manager.load_labels_from_jsonl()
manager.validate_structure()
""")

    print("\n2. LOAD 5% FOR TESTING (< 30 sec, ~1.6 GB):")
    print("""
from load_dataset_optimized import CPGDatasetManagerOptimized
from torch.utils.data import DataLoader

manager = CPGDatasetManagerOptimized()
val_dataset, metadata = manager.load_and_validate(fraction=0.05)

loader = DataLoader(val_dataset, batch_size=16, num_workers=0)
for batch in loader:
    # Process batch
    print(f"Batch loaded")
    break
""")

    print("\n3. CREATE MULTIPLE SUBSETS (Memory efficient):")
    print("""
manager = CPGDatasetManagerOptimized()
manager.load_split_data()
manager.load_labels_from_jsonl()

for i in range(5):
    indices = manager.create_subset_indices(fraction=0.1)
    dataset = CPGDatasetOptimized(
        tensor_path=...,
        indices=indices,
        labels_dict=manager.labels,
        id_map=manager.id_map
    )
    # Process dataset
    del dataset
    gc.collect()
""")


def main():
    """Run all strategies demonstration"""
    print("\n" + "█"*80)
    print("█ CPU-OPTIMIZED DATASET LOADING STRATEGIES")
    print("█"*80)

    # Show comparison
    strategy_5_comparison()

    # Show code snippets
    get_quick_start_code()

    print("\n" + "█"*80)
    print("█ RECOMMENDED NEXT STEPS")
    print("█"*80)
    print("""
1. Run the optimized loader:
   python load_dataset_optimized.py

2. Check memory usage during loading:
   - Monitor with: Task Manager (Windows) or 'top' (Linux)

3. Test your model on small fraction:
   - Start with 5% for quick iterations
   - Increase to 25% for validation

4. For production:
   - Use full dataset or appropriate fraction
   - Consider using DataLoader num_workers for parallelism
""")


if __name__ == "__main__":
    main()
