# Google Colab GPU Setup Guide

## Quick Start (Recommended)

### Step 1: Enable GPU in Colab
1. In Google Colab, click **Runtime** → **Change runtime type**
2. Select **GPU** as the hardware accelerator
3. Click **Save**

### Step 2: Paste the Quickstart Code
Copy the entire contents of `colab_quickstart.py` and paste into a Colab cell.

### Step 3: Configure (Optional)
Edit these variables at the top of the cell:

```python
DATASET_PATH = "/content/drive/MyDrive/CPG"      # Your dataset location
PROCESSED_DIR = "processed_homo"                  # or "processed_hetero"
FRACTION = 0.05                                   # 0.05 = 5%, 1.0 = 100%
BATCH_SIZE = 16
DEVICE = "auto"                                   # "auto", "cuda", or "cpu"
```

### Step 4: Run
Execute the cell. The script will:
- Auto-detect GPU if available
- Load the dataset
- Show device info
- Test first batch

---

## Device Options

### `DEVICE = "auto"` (Recommended)
- Automatically uses GPU if available
- Falls back to CPU if GPU not available
- Best for most cases

### `DEVICE = "cuda"` (Force GPU)
- Always use GPU
- Will give error if GPU not available
- Use when you need GPU-specific optimizations

### `DEVICE = "cpu"` (Force CPU)
- Always use CPU
- Useful for testing or debugging
- Slower than GPU but less memory

---

## Checking GPU Status

Add this to any Colab cell to check GPU:

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"PyTorch Version: {torch.__version__}")
```

---

## Advanced: Custom Device Configuration

```python
# Force CPU (useful for testing)
from load_dataset_configurable import CPGDatasetManagerConfigurable

manager = CPGDatasetManagerConfigurable(device="cpu")
dataset, meta = manager.load_and_validate(fraction=0.05)

# Force GPU
manager = CPGDatasetManagerConfigurable(device="cuda")
dataset, meta = manager.load_and_validate(fraction=0.25)

# Auto-detect (default)
manager = CPGDatasetManagerConfigurable(device="auto")
dataset, meta = manager.load_and_validate(fraction=0.1)
```

---

## Memory Considerations

### GPU Memory (Colab typically has 15-16 GB)
- **5% dataset on GPU**: ~8 GB
- **10% dataset on GPU**: ~16 GB (might be tight)
- **25% dataset on GPU**: Might exceed memory

### CPU Memory (Colab has 13 GB)
- **5% dataset on CPU**: ~2 GB
- **10% dataset on CPU**: ~4 GB
- **25% dataset on CPU**: ~8 GB
- **50% dataset on CPU**: ~16 GB (might exceed)

### Recommendation
Use GPU for training (faster), adjust fraction as needed.

---

## Moving Data to GPU During Training

If you load on CPU but want to train on GPU:

```python
loader, metadata, dataset = load_dataset_colab(
    dataset_path=DATASET_PATH,
    device="cpu"  # Load on CPU
)

# In training loop
for batch in loader:
    graph, label, idx, name = batch
    # Move to GPU if needed
    if torch.cuda.is_available():
        graph = graph.to("cuda")
        label = label.to("cuda")

    # Train
    output = model(graph)
    loss = criterion(output, label)
```

---

## Troubleshooting

### GPU not available in Colab
- Check: Runtime → Change runtime type → Select GPU
- Colab has limited GPU quota, sometimes not available

### CUDA out of memory
- Reduce `FRACTION` (e.g., 0.05 instead of 0.25)
- Reduce `BATCH_SIZE` (e.g., 8 instead of 16)
- Use CPU instead: `DEVICE = "cpu"`

### Slow training on GPU
- Check GPU is actually being used: `torch.cuda.current_device()`
- Move data to GPU before processing
- Ensure model is on GPU: `model.to("cuda")`

---

## Full Training Loop Example

```python
# Load with automatic device selection
device = "auto"  # or "cuda" or "cpu"

loader, metadata, dataset = load_dataset_colab(
    dataset_path=DATASET_PATH,
    fraction=0.05,
    device=device
)

print(f"Training on: {device}")

# Your model
model = YourGNNModel()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch in loader:
        graph, label, idx, name = batch

        # Move to device
        graph = graph.to(device)
        label = label.to(device)

        # Forward
        output = model(graph)
        loss = criterion(output, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

For more info, check `README_DATASET_LOADING.md` and `CONFIGURATION_GUIDE.py`
