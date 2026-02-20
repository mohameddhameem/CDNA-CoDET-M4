"""
Explore the 2way_base_split_target.json file to understand the dataset split
"""

import json
from pathlib import Path
from collections import Counter

def main():
    json_file = Path(r"c:\Learning\SMU\City-of-Agents-1\dataset\data_split\2way_base_split_target.json")

    print("="*70)
    print("EXPLORING: 2way_base_split_target.json")
    print("="*70)

    with open(json_file, 'r') as f:
        split_data = json.load(f)

    # Extract data
    label_map = split_data['label_map']  # {'ai': 0, 'human': 1}
    id_map = split_data['id_map']  # {'0': 'ai', '1': 'human'}
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']

    print("\n1. LABEL MAPPING")
    print("-" * 70)
    print(f"   Label to Index: {label_map}")
    print(f"   Index to Label: {id_map}")

    print("\n2. DATASET SPLIT STATISTICS")
    print("-" * 70)
    total_samples = len(train_indices) + len(test_indices)
    train_pct = (len(train_indices) / total_samples) * 100
    test_pct = (len(test_indices) / total_samples) * 100

    print(f"   Train Set: {len(train_indices):,} samples ({train_pct:.1f}%)")
    print(f"   Test Set:  {len(test_indices):,} samples ({test_pct:.1f}%)")
    print(f"   Total:     {total_samples:,} samples")

    print("\n3. INDEX RANGES")
    print("-" * 70)
    print(f"   Train indices range: {min(train_indices)} to {max(train_indices)}")
    print(f"   Test indices range:  {min(test_indices)} to {max(test_indices)}")
    print(f"   Overall range:       {min(train_indices + test_indices)} to {max(train_indices + test_indices)}")

    print("\n4. DATA INTEGRITY CHECK")
    print("-" * 70)

    # Check for overlaps
    train_set = set(train_indices)
    test_set = set(test_indices)
    overlap = train_set & test_set

    print(f"   Overlap between train and test: {len(overlap)} (should be 0)")
    if overlap:
        print(f"   WARNING: Found {len(overlap)} overlapping indices!")

    # Check for duplicates
    train_dupes = len(train_indices) - len(train_set)
    test_dupes = len(test_indices) - len(test_set)
    print(f"   Duplicates in train: {train_dupes}")
    print(f"   Duplicates in test: {test_dupes}")

    # Check continuity
    all_indices = sorted(train_indices + test_indices)
    expected_indices = set(range(total_samples))
    actual_indices = set(all_indices)
    missing = expected_indices - actual_indices

    if missing:
        print(f"   WARNING: Found {len(missing)} missing indices: {sorted(list(missing))[:10]}...")
    else:
        print(f"   ✓ All indices from 0 to {total_samples-1} are present")

    print("\n5. SAMPLE INDICES")
    print("-" * 70)
    print(f"   First 10 train indices: {train_indices[:10]}")
    print(f"   First 10 test indices:  {test_indices[:10]}")
    print(f"   Last 5 train indices:   {train_indices[-5:]}")
    print(f"   Last 5 test indices:    {test_indices[-5:]}")

    print("\n6. DISTRIBUTION INFORMATION")
    print("-" * 70)
    print(f"   To determine label distribution, we need to:")
    print(f"   - Load the original JSONL file with labels")
    print(f"   - Map these indices to their labels")
    print(f"   - Count class distribution in train/test splits")
    print(f"\n   Dataset location: dataset/cpg/cpg_dataset_raw10.jsonl")

    print("\n" + "="*70)
    print("READY TO USE: CPGDatasetManager.load_all()")
    print("="*70)
    print("\nThe dataset is now ready to be loaded with the full load_dataset.py")
    print("which will:")
    print("  1. Load processed.pt (graph tensors)")
    print("  2. Load labels from JSONL")
    print("  3. Use these indices for train/test split")
    print("  4. Create PyTorch datasets with proper labels")


if __name__ == "__main__":
    main()
