"""
Complete Dataset Exploration Script
Analyzes the JSON split file and provides dataset statistics
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_json_split(json_path: str) -> Dict:
    """Read and parse the JSON split file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_split_structure(split_data: Dict) -> None:
    """Analyze the structure of the split data"""
    print("\n" + "="*80)
    print("SPLIT DATA STRUCTURE")
    print("="*80)

    label_map = split_data['label_map']
    id_map = split_data['id_map']
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']

    print(f"\nLabel Mapping:")
    print(f"  {label_map}")
    print(f"\nID Mapping (reverse):")
    print(f"  {id_map}")

    return label_map, id_map, train_indices, test_indices


def analyze_split_sizes(train_indices: List[int], test_indices: List[int]) -> None:
    """Analyze train/test split sizes"""
    print("\n" + "="*80)
    print("DATASET SPLIT SIZES")
    print("="*80)

    total = len(train_indices) + len(test_indices)
    train_pct = (len(train_indices) / total) * 100
    test_pct = (len(test_indices) / total) * 100

    print(f"\nTrain Set:")
    print(f"  Size: {len(train_indices):,} samples")
    print(f"  Percentage: {train_pct:.2f}%")

    print(f"\nTest Set:")
    print(f"  Size: {len(test_indices):,} samples")
    print(f"  Percentage: {test_pct:.2f}%")

    print(f"\nTotal Samples: {total:,}")


def analyze_index_ranges(train_indices: List[int], test_indices: List[int]) -> None:
    """Analyze the range of indices"""
    print("\n" + "="*80)
    print("INDEX RANGES")
    print("="*80)

    all_indices = train_indices + test_indices
    min_idx = min(all_indices)
    max_idx = max(all_indices)

    print(f"\nTrain Index Range:")
    print(f"  Min: {min(train_indices)}")
    print(f"  Max: {max(train_indices)}")

    print(f"\nTest Index Range:")
    print(f"  Min: {min(test_indices)}")
    print(f"  Max: {max(test_indices)}")

    print(f"\nOverall Range: {min_idx} to {max_idx}")
    print(f"  Span: {max_idx - min_idx + 1}")


def check_data_integrity(train_indices: List[int], test_indices: List[int]) -> None:
    """Check for data integrity issues"""
    print("\n" + "="*80)
    print("DATA INTEGRITY CHECK")
    print("="*80)

    train_set = set(train_indices)
    test_set = set(test_indices)

    # Check overlaps
    overlap = train_set & test_set
    print(f"\n✓ Train/Test Overlap: {len(overlap)} indices (should be 0)")
    if overlap:
        print(f"  WARNING: Found overlapping indices: {list(overlap)[:5]}...")

    # Check duplicates
    train_dupes = len(train_indices) - len(train_set)
    test_dupes = len(test_indices) - len(test_set)
    print(f"✓ Train Duplicates: {train_dupes}")
    print(f"✓ Test Duplicates: {test_dupes}")

    # Check if indices are continuous
    total_samples = len(train_indices) + len(test_indices)
    expected_indices = set(range(total_samples))
    actual_indices = train_set | test_set
    missing = expected_indices - actual_indices

    print(f"✓ Missing Indices: {len(missing)}")
    if missing:
        missing_list = sorted(list(missing))
        if len(missing_list) > 10:
            print(f"  First 10 missing: {missing_list[:10]}")
        else:
            print(f"  Missing indices: {missing_list}")


def show_sample_indices(train_indices: List[int], test_indices: List[int], n: int = 10) -> None:
    """Show sample of indices from both sets"""
    print("\n" + "="*80)
    print(f"SAMPLE INDICES (first {n} from each set)")
    print("="*80)

    print(f"\nTrain Indices (first {n}):")
    for i, idx in enumerate(train_indices[:n], 1):
        print(f"  {i:2d}. {idx}")

    print(f"\nTest Indices (first {n}):")
    for i, idx in enumerate(test_indices[:n], 1):
        print(f"  {i:2d}. {idx}")

    print(f"\nTrain Indices (last {n}):")
    for i, idx in enumerate(train_indices[-n:], 1):
        print(f"  {i:2d}. {idx}")

    print(f"\nTest Indices (last {n}):")
    for i, idx in enumerate(test_indices[-n:], 1):
        print(f"  {i:2d}. {idx}")


def check_files_exist() -> None:
    """Check if required files exist"""
    print("\n" + "="*80)
    print("FILE AVAILABILITY CHECK")
    print("="*80)

    base_path = Path(r"c:\Learning\SMU\City-of-Agents-1")

    files_to_check = [
        ("Tensor Data", base_path / "dataset" / "processed_homo" / "processed.pt"),
        ("Pre-filter Data", base_path / "dataset" / "processed_homo" / "pre_filter.pt"),
        ("Pre-transform Data", base_path / "dataset" / "processed_homo" / "pre_transform.pt"),
        ("Raw JSONL", base_path / "dataset" / "cpg" / "cpg_dataset_raw10.jsonl"),
        ("Split JSON", base_path / "dataset" / "data_split" / "2way_base_split_target.json"),
    ]

    print()
    for name, path in files_to_check:
        exists = path.exists()
        status = "✓" if exists else "✗"
        size = f" ({path.stat().st_size / (1024**3):.2f} GB)" if exists and path.is_file() else ""
        print(f"{status} {name:20s}: {path}{size}")


def main():
    """Main exploration function"""
    json_path = r"c:\Learning\SMU\City-of-Agents-1\dataset\data_split\2way_base_split_target.json"

    print("\n" + "█"*80)
    print("█ COMPREHENSIVE DATASET EXPLORATION: 2way_base_split_target.json")
    print("█"*80)

    # Check files
    check_files_exist()

    # Read JSON
    split_data = read_json_split(json_path)

    # Analyze structure
    label_map, id_map, train_indices, test_indices = analyze_split_structure(split_data)

    # Analyze sizes
    analyze_split_sizes(train_indices, test_indices)

    # Analyze ranges
    analyze_index_ranges(train_indices, test_indices)

    # Check integrity
    check_data_integrity(train_indices, test_indices)

    # Show samples
    show_sample_indices(train_indices, test_indices, n=10)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & NEXT STEPS")
    print("="*80)
    print(f"""
This JSON file defines:
  • Label mapping: {label_map}
  • Train/Test split: {len(train_indices):,} / {len(test_indices):,} samples
  • Total dataset size: {len(train_indices) + len(test_indices):,} graphs

Next Steps:
  1. Use load_dataset.py to load the tensor files
  2. Match indices with labels from cpg_dataset_raw10.jsonl
  3. Create PyTorch Dataset with proper train/test split
  4. Analyze class distribution (AI vs Human)
  5. Build and train your model

Command to run:
  python load_dataset.py
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
