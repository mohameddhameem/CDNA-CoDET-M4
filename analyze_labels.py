"""
Load and Analyze Labels with Split Information
Maps indices to labels and shows class distribution
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def load_jsonl_labels(jsonl_path: str, label_map: Dict[str, int]) -> Dict[int, Tuple[str, int]]:
    """
    Load labels from JSONL file and map to indices.

    Args:
        jsonl_path: Path to the JSONL file
        label_map: Dictionary mapping label names to indices

    Returns:
        Dictionary mapping sample index to (label_name, label_idx)
    """
    labels = {}
    print(f"Loading labels from: {jsonl_path}")

    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            target = record.get('target', 'unknown')
            label_idx = label_map.get(target, -1)
            labels[idx] = (target, label_idx)

            if (idx + 1) % 50000 == 0:
                print(f"  Loaded {idx + 1} labels...")

    print(f"  Total labels loaded: {len(labels)}")
    return labels


def analyze_class_distribution(indices: List[int], labels: Dict[int, Tuple[str, int]], split_name: str) -> None:
    """Analyze class distribution for a given set of indices"""
    label_names = []
    label_counts = Counter()

    for idx in indices:
        if idx in labels:
            label_name, _ = labels[idx]
            label_names.append(label_name)
            label_counts[label_name] += 1

    print(f"\n{split_name}:")
    print(f"  Total samples: {len(indices)}")

    for label_name in sorted(label_counts.keys()):
        count = label_counts[label_name]
        percentage = (count / len(indices)) * 100
        print(f"  {label_name:10s}: {count:7,} ({percentage:6.2f}%)")


def main():
    """Main function to analyze labels with split"""
    base_path = Path(r"c:\Learning\SMU\City-of-Agents-1")

    jsonl_path = base_path / "dataset" / "cpg" / "cpg_dataset_raw10.jsonl"
    json_split_path = base_path / "dataset" / "data_split" / "2way_base_split_target.json"

    print("\n" + "█"*80)
    print("█ LABEL DISTRIBUTION ANALYSIS")
    print("█"*80)

    # Load split data
    print(f"\nLoading split from: {json_split_path}")
    with open(json_split_path, 'r') as f:
        split_data = json.load(f)

    label_map = split_data['label_map']
    id_map = split_data['id_map']
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']

    print(f"  Label map: {label_map}")
    print(f"  Train indices: {len(train_indices)}")
    print(f"  Test indices: {len(test_indices)}")

    # Load labels from JSONL
    labels = load_jsonl_labels(str(jsonl_path), label_map)

    # Analyze distribution
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION")
    print("="*80)

    analyze_class_distribution(train_indices, labels, "TRAIN SET")
    analyze_class_distribution(test_indices, labels, "TEST SET")

    # Overall statistics
    print(f"\nOVERALL:")
    all_indices = train_indices + test_indices
    overall_counts = Counter()

    for idx in all_indices:
        if idx in labels:
            label_name, _ = labels[idx]
            overall_counts[label_name] += 1

    for label_name in sorted(overall_counts.keys()):
        count = overall_counts[label_name]
        percentage = (count / len(all_indices)) * 100
        print(f"  {label_name:10s}: {count:7,} ({percentage:6.2f}%)")

    # Show sample records
    print("\n" + "="*80)
    print("SAMPLE RECORDS")
    print("="*80)

    print(f"\nFirst 5 train samples:")
    for i, idx in enumerate(train_indices[:5], 1):
        if idx in labels:
            label_name, label_idx = labels[idx]
            print(f"  {i}. Index {idx:6d} -> Label: {label_name} (idx: {label_idx})")

    print(f"\nFirst 5 test samples:")
    for i, idx in enumerate(test_indices[:5], 1):
        if idx in labels:
            label_name, label_idx = labels[idx]
            print(f"  {i}. Index {idx:6d} -> Label: {label_name} (idx: {label_idx})")

    print("\n" + "="*80)
    print("✓ Dataset is ready for model building!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
