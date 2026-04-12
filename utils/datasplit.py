import os
import json
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

def process_datasplit(idx_list, label_list, output_dir='./data_split', n_tasks=50, max_shot=10, task_name='target'):
    """
    Args:
        idx_list (list): List of data indices / identifiers.
        label_list (list): List of corresponding labels (str or int).
        output_dir (str): Directory to save the JSON outputs.
        n_tasks (int): Number of few-shot tasks to sample.
        max_shot (int): Maximum shot number.
        task_name (str): Name of the current task (e.g., 'target', 'model').
    """
    
    # 1. Pre-processing: Ensure inputs are Python Lists (not Tensors)
    if isinstance(idx_list, torch.Tensor):
        idx_list = idx_list.tolist()
    if isinstance(label_list, torch.Tensor):
        label_list = label_list.tolist()

    # 2. Basic Assertions
    assert len(idx_list) == len(label_list), f"Length mismatch: Indices {len(idx_list)} vs Labels {len(label_list)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n[{task_name.upper()}] Processing {len(idx_list)} samples...")

    # ==========================================
    # Step 1: Label Encoding (String -> Int)
    # ==========================================
    # Get unique labels and sort them for determinism
    # Handle potential None/NaN values by converting to string
    unique_labels = sorted(list(set(str(l) for l in label_list)))
    n_way = len(unique_labels)
    
    # Create mapping dictionaries
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    print(f"Total Classes: {n_way}")
    
    # Convert string labels to integer labels for splitting
    # Ensure we use the string representation to match the key in label_to_id
    label_list_int = [label_to_id[str(label)] for label in label_list]

    # ==========================================
    # Step 2: Train/Test Split (20:80)
    # ==========================================
    # Stratify ensures consistent class distribution
    train_idxs, test_idxs, train_labels, test_labels = train_test_split(
        idx_list, 
        label_list_int, 
        train_size=0.2, 
        test_size=0.8, 
        stratify=label_list_int, 
        random_state=42 
    )

    print(f"Split Complete -> Train: {len(train_idxs)}, Test: {len(test_idxs)}")

    # ==========================================
    # Step 3: Organize Training Data
    # ==========================================
    # Structure: { class_id (int): [idx1, idx2, ...] }
    class_to_indices = defaultdict(list)
    for idx, label_id in zip(train_idxs, train_labels):
        class_to_indices[label_id].append(idx)

    # Filter valid classes (must have at least 'max_shot' samples)
    valid_classes = []
    for label_id, indices in class_to_indices.items():
        if len(indices) >= max_shot:
            valid_classes.append(label_id)
        
    print(f"Valid classes for {max_shot}-shot sampling: {len(valid_classes)} / {n_way}")

    if len(valid_classes) == 0:
        print("Error: No classes have enough samples for sampling. Exiting this task.")
        return

    # ==========================================
    # Step 4: Nested Few-Shot Sampling
    # ==========================================
    all_tasks = []
    print(f"Sampling {n_tasks} tasks...")
    
    for task_id in tqdm(range(n_tasks), desc=f"Sampling {task_name}"):
        # Structure: {'1_shot': {cls_id: [idx]}, ...}
        task_data = defaultdict(dict)
        
        # Iterate over every valid class
        for label_id in valid_classes:
            pool = class_to_indices[label_id]
            
            # --- Nested Sampling Logic ---
            # 1. Sample 'max_shot' unique items
            selected_samples = random.sample(pool, max_shot)
            
            # 2. Distribute via slicing (Inclusive/Nested)
            for k in range(1, max_shot + 1):
                # Convert label_id to string for JSON compatibility
                task_data[f"{k}_shot"][str(label_id)] = selected_samples[:k]
        
        all_tasks.append({
            "task_id": task_id,
            "data": task_data
        })

    # ==========================================
    # Step 5: Save Results
    # ==========================================
    
    # 5.1 Save Base Split
    split_info = {
        "label_map": label_to_id,       
        "id_map": id_to_label,          
        "train_indices": train_idxs,
        "test_indices": test_idxs
    }
    
    # Use task_name in filename
    split_filename = f"{n_way}way_base_split_{task_name}.json"
    split_path = os.path.join(output_dir, split_filename)
    
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=4)
    
    # 5.2 Save tasks
    task_filename = f"{n_way}way_{n_tasks}tasks_{task_name}.json"
    task_path = os.path.join(output_dir, task_filename)
    
    with open(task_path, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=4)

    print(f"Saved: {split_filename}")
    print(f"Saved: {task_filename}")


if __name__ == "__main__":
    from cpg2hetero import CPGHeteroDataset

    # 1. Load Dataset ONCE
    print("Loading Dataset...")
    dataset = CPGHeteroDataset(root='./CPG', force_reload=False)
    
    # Extract common indices (assuming they are the same for both tasks)
    # Ensure it's a list
    all_indices = dataset.data.dataset_idx
    if isinstance(all_indices, torch.Tensor):
        all_indices = all_indices.tolist()

    # 2. Define Tasks config
    # We iterate through this list to avoid code duplication
    tasks = [
        ('target', dataset.data.target),
        ('model',  dataset.data.model)
    ]

    # 3. Process each task
    for task_name, label_data in tasks:
        # Convert labels to list if they are tensors
        if isinstance(label_data, torch.Tensor):
            labels = label_data.tolist()
        else:
            labels = list(label_data)

        # Run the processor
        process_datasplit(
            idx_list=all_indices, 
            label_list=labels, 
            task_name=task_name,
            output_dir='./CPG/data_split',
        )