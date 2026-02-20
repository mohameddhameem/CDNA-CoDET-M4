# Colab Quick Setup

## 1. Enable GPU (1 minute)

In Google Colab:
1. **Runtime** → **Change runtime type**
2. Select **GPU** (or **TPU** for faster training)
3. Click **Save**

## 2. Check GPU (run in Colab cell)

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
```

## 3. Mount Drive (run in first cell)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. Load Dataset (copy-paste entire code)

```python
# Copy ENTIRE content of colab_quickstart.py
# Paste into Colab cell
# Edit top variables:
DATASET_PATH = "/content/drive/MyDrive/CPG"
DEVICE = "auto"  # or "cuda" or "cpu"
FRACTION = 0.05
```

## Device Options

| Option | Use Case |
|--------|----------|
| `"auto"` | Recommended - uses GPU if available |
| `"cuda"` | Force GPU (error if unavailable) |
| `"cpu"` | Force CPU (slower but uses less memory) |

## Expected Output

```
Device: CUDA
GPU: Tesla T4 (or similar)
GPU Memory: 15.1 GB
✓ Loaded 12,000 samples | Classes: {'ai': 6000, 'human': 6000}
  DataLoader: 750 batches of 16
✓ First batch loaded | 4 items
✓ Ready for training!
```

## For Training

```python
# Tensors load to GPU automatically
for batch in loader:
    graph, label, idx, name = batch
    # graph is already on GPU (or CPU, depending on DEVICE)
    output = model(graph)
    loss = criterion(output, label)
    loss.backward()
```

## Memory Limits

**GPU (15 GB available):**
- 5% = 8 GB ✓
- 10% = 16 GB ⚠️
- 25%+ = Too much

**CPU (13 GB available):**
- 5% = 2 GB ✓
- 10% = 4 GB ✓
- 25% = 8 GB ✓
- 50% = 16 GB ⚠️

## Troubleshooting

**GPU not showing?**
→ Runtime → Change runtime type → Select GPU

**Running out of memory?**
→ Reduce `FRACTION` or use CPU

**Still have issues?**
→ Check `COLAB_GPU_SETUP.md` for detailed guide

---

**That's it! You're ready to train.** 🚀
