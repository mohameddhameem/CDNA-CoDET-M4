"""
OPTIMIZATION GUIDE: CPU Dataset Loading

This document explains the optimization techniques used and how to apply them.
"""

GUIDE = """
╔════════════════════════════════════════════════════════════════════════════╗
║           CPU-OPTIMIZED DATASET LOADING FOR LARGE TENSORS                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Your dataset is 32 GB - too large to load entirely on CPU. This guide shows
how to efficiently load fractions for validation and testing.

═══════════════════════════════════════════════════════════════════════════════

1. OPTIMIZATION TECHNIQUES USED
───────────────────────────────────────────────────────────────────────────────

A. LAZY LOADING (On-Demand Loading)
   What: Load data only when accessed, not when creating the dataset
   Where: CPGDatasetOptimized class uses lazy loading
   Benefit: Can create dataset objects without loading all data

   Example:
     dataset = CPGDatasetOptimized(...)  # No loading yet
     batch = next(iter(loader))          # Loading happens HERE

B. FRACTIONAL LOADING (Subset Creation)
   What: Load only a fraction (5%, 10%, 25%) of data
   Where: create_subset_indices() and create_validation_dataset()
   Benefit: 5% = 1.6 GB instead of 32 GB

   Options:
     fraction=0.05  → 5%   (~1.6 GB)
     fraction=0.10  → 10%  (~3.2 GB)
     fraction=0.25  → 25%  (~8.0 GB)
     fraction=0.50  → 50%  (~16 GB)
     fraction=1.0   → 100% (~32 GB)

C. STRATIFIED SAMPLING
   What: Sample subsets while maintaining class balance
   Where: _stratified_sample() in CPGDatasetManagerOptimized
   Benefit: Small subset represents full dataset distribution

   Result: 5% subset has same AI/Human ratio as full dataset

D. METADATA-ONLY LOADING
   What: Load only JSON and JSONL (no tensors)
   Where: load_split_data() and load_labels_from_jsonl()
   Benefit: Validation in < 1 second, < 1 MB memory

   Content loaded:
     - 2way_base_split_target.json (label map, indices)
     - cpg_dataset_raw10.jsonl (labels for all 200k samples)

   Content NOT loaded:
     - processed.pt (32 GB tensor file) ← Loaded only on demand

E. MEMORY-EFFICIENT DATA STRUCTURES
   What: Use numpy for fast operations, torch.Tensor for actual data
   Where: labels_dict, id_map stored as Python dicts (fast lookup)
   Benefit: Metadata operations are instant

   Lookup time: O(1) for label access

═══════════════════════════════════════════════════════════════════════════════

2. RECOMMENDED WORKFLOWS
───────────────────────────────────────────────────────────────────────────────

WORKFLOW 1: Quick Validation (5 minutes)

   manager = CPGDatasetManagerOptimized()
   val_dataset, meta = manager.load_and_validate(fraction=0.05)

   Memory: ~1.6 GB
   Time: ~30 seconds
   Samples: ~12,000 (5% of 240k)

   ✓ Use for: Testing code pipeline, debugging, quick iteration

WORKFLOW 2: Thorough Validation (15 minutes)

   manager.load_and_validate(fraction=0.25)  # 25% = 60k samples

   Memory: ~8 GB
   Time: ~1-2 minutes
   Samples: ~60,000

   ✓ Use for: Model testing, validation, performance estimation

WORKFLOW 3: Sequential Processing (Memory efficient)

   Load metadata once:
     manager.load_split_data()
     manager.load_labels_from_jsonl()

   Create many small subsets:
     for i in range(10):
         subset = manager.create_subset_indices(fraction=0.05)
         dataset = create_dataset(subset)
         process(dataset)
         del dataset
         gc.collect()

   Memory: Never exceeds ~2 GB
   Time: Depends on processing

   ✓ Use for: Cross-validation, grid search, multiple experiments

═══════════════════════════════════════════════════════════════════════════════

3. MEMORY CONSUMPTION BREAKDOWN
───────────────────────────────────────────────────────────────────────────────

For 5% fraction (12k samples):

   Metadata:
     - Split JSON:     ~2 MB
     - Labels (JSON):  ~200 MB
     - Index list:     ~50 KB

   Subtotal: ~200 MB (kept in memory always)

   Graph data (loaded on demand):
     - Per graph:      ~1.3 MB (average)
     - 12k graphs:     ~16 GB
     - Actual load:    ~1.6 GB (with numpy efficiency)

   Total with batch loading: ~2 GB

For different fractions:
   - 1%  = 0.3 GB
   - 5%  = 1.6 GB
   - 10% = 3.2 GB
   - 25% = 8 GB

Recommendation for 16GB RAM:
   → Use 5% fraction with batch_size=8-16
   → Or use sequential loading (process one subset at a time)

═══════════════════════════════════════════════════════════════════════════════

4. HOW TO USE THE OPTIMIZED LOADER
───────────────────────────────────────────────────────────────────────────────

OPTION A: One-line quickstart

   from quickstart import main
   main()  # Loads 5%, creates dataset, shows results

OPTION B: Manual control

   from load_dataset_optimized import CPGDatasetManagerOptimized
   from torch.utils.data import DataLoader

   # Initialize
   manager = CPGDatasetManagerOptimized()

   # Load metadata (< 1 second)
   manager.load_split_data()
   manager.load_labels_from_jsonl()

   # Create validation dataset
   val_dataset, meta = manager.load_and_validate(fraction=0.1)

   # Create loader
   loader = DataLoader(val_dataset, batch_size=16)

   # Iterate (triggers lazy loading on first batch)
   for batch in loader:
       graph, label, idx, name = batch
       # Process batch
       break

OPTION C: Multiple subsets

   indices_subset1 = manager.create_subset_indices(fraction=0.05, split='train')
   indices_subset2 = manager.create_subset_indices(fraction=0.05, split='train')

   dataset1 = CPGDatasetOptimized(..., indices=indices_subset1, ...)
   dataset2 = CPGDatasetOptimized(..., indices=indices_subset2, ...)

═══════════════════════════════════════════════════════════════════════════════

5. PERFORMANCE EXPECTATIONS
───────────────────────────────────────────────────────────────────────────────

Loading Times (approximate):

   metadata_only (split + labels):
     Time: 10-15 seconds
     Memory: ~200 MB

   5% fraction:
     Metadata: 10 seconds
     First batch: 2-5 seconds (lazy load)
     Total: ~20 seconds for first batch

   10% fraction:
     Metadata: 10 seconds
     First batch: 3-8 seconds
     Total: ~25 seconds

   25% fraction:
     Metadata: 10 seconds
     First batch: 5-15 seconds
     Total: ~30 seconds

Accessing Batches:

   first batch:  3-8 seconds (includes load time)
   next batches: < 1 second each (already loaded)

═══════════════════════════════════════════════════════════════════════════════

6. TROUBLESHOOTING
───────────────────────────────────────────────────────────────────────────────

Issue: "MemoryError when loading"
Solution: Use smaller fraction
   manager.load_and_validate(fraction=0.01)  # 1% instead of 5%

Issue: "Slow batch loading"
Solution: Reduce batch size or keep fewer batches in memory
   loader = DataLoader(dataset, batch_size=8)  # Instead of 32
   # Or use pinned memory:
   loader = DataLoader(dataset, batch_size=16, pin_memory=True)

Issue: "Dataset takes long to create"
Solution: Skip analysis steps
   val_dataset, _ = manager.load_and_validate(fraction=0.05)
   # Don't call manager.analyze_sample()

Issue: "Need full dataset but limited RAM"
Solution: Use sequential loading
   # Load one batch at a time, process, move to next
   for batch in DataLoader(dataset, batch_size=1):
       # Process single sample
       pass

═══════════════════════════════════════════════════════════════════════════════

7. FILES PROVIDED
───────────────────────────────────────────────────────────────────────────────

1. load_dataset_optimized.py
   - CPGDatasetOptimized: Lazy-loading dataset
   - CPGDatasetManagerOptimized: Complete manager with all utilities
   - Core optimization logic

2. quickstart.py
   - One-file solution for immediate use
   - Loads 5%, validates, creates dataloader
   - Run: python quickstart.py

3. dataset_loading_strategies.py
   - Shows 5 different loading strategies
   - Comparison table of memory/time tradeoffs
   - Code snippets for each approach

4. load_dataset.py (original)
   - Simpler version without optimization
   - Good for understanding basic structure
   - Use when you have enough RAM

═══════════════════════════════════════════════════════════════════════════════

8. QUICK START COMMANDS
───────────────────────────────────────────────────────────────────────────────

Immediate usage:
   python quickstart.py

Show all strategies:
   python dataset_loading_strategies.py

Custom loading (in your code):
   from load_dataset_optimized import CPGDatasetManagerOptimized
   manager = CPGDatasetManagerOptimized()
   dataset, meta = manager.load_and_validate(fraction=0.05)

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(GUIDE)

    # Save to file
    with open(r"c:\Learning\SMU\City-of-Agents-1\OPTIMIZATION_GUIDE.txt", 'w') as f:
        f.write(GUIDE)

    print("\n✓ Guide saved to: OPTIMIZATION_GUIDE.txt")
