# HPC Deployment Guide

## Overview
This guide explains how to deploy and run the CPG dataset generation pipeline on an HPC cluster.

## Prerequisites
- JDK installed on HPC (Java 8+)
- Python 3.8+ with required packages
- joern-cli directory copied to HPC
- Project code copied to HPC

## Step 1: Prepare HPC Environment

### 1a. Copy Required Resources
```bash
# On your local machine, create a bundle with JDK and joern-cli
# Then transfer to HPC (adjust paths as needed)
scp -r /path/to/jdk-19 hpc:/shared/software/
scp -r /path/to/joern-cli hpc:/shared/software/
scp -r /path/to/City-of-Agents-1 hpc:~/projects/
```

### 1b. Load/Configure Java (if not already set up)

For bash/sh:
```bash
export JAVA_HOME=/shared/software/jdk-19
export PATH=$JAVA_HOME/bin:$PATH
```

Verify Java works:
```bash
java -version  # Should output Java version info
```

### 1c. Setup Python Environment

```bash
# Load Python module (adjust based on your HPC)
module load python/3.11

# Or activate conda environment
conda activate py311

# Verify required packages are installed
pip install datasets transformers torch torch-geometric tqdm
```

## Step 2: Submit Generation Jobs

### Option 1: Sequential Generation (Recommended for first run)

```bash
cd ~/projects/City-of-Agents-1

# Step 1: Generate CPGs with Joern (takes longest)
export JOERN_PATH=/shared/software/joern-cli
export WORKERS=32  # Adjust based on HPC resources
bash scripts/generate_joern.sh
# Wait for completion, check with: tail -f logs/Joern/Joern_generate_*.log

# Step 2: Generate Heterogeneous Graphs
export WORKERS=64
bash scripts/generate_hetero.sh
# Wait for completion

# Step 3: Generate Homogeneous Graphs
export WORKERS=64
bash scripts/generate_homo.sh
```

### Option 2: Using HPC Job Submission (SLURM example)

Create `run_pipeline.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=cpg-pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/pipeline_%j.log

# Load required modules
module load java/19
module load python/3.11

# Set up environment
cd ~/projects/City-of-Agents-1
export JOERN_PATH=/shared/software/joern-cli
export WORKERS=64
export CUDA_DEVICE=0,1,2,3

# Step 1: CPG Generation (Joern)
echo "Starting Joern CPG generation..."
bash scripts/generate_joern.sh
JOB1_PID=$(cat pids/Joern_*.pid | tail -1)
wait $JOB1_PID

# Step 2: Heterogeneous Graph Processing
echo "Starting heterogeneous graph generation..."
bash scripts/generate_hetero.sh
JOB2_PID=$(cat pids/Hetero_*.pid | tail -1)
wait $JOB2_PID

# Step 3: Homogeneous Graph Processing
echo "Starting homogeneous graph generation..."
bash scripts/generate_homo.sh
JOB3_PID=$(cat pids/Homo_*.pid | tail -1)
wait $JOB3_PID

echo "Pipeline completed!"
```

Submit the job:
```bash
sbatch run_pipeline.sh
```

### Option 3: Parallel Jobs (faster if resources available)

For large HPC clusters, you can run correlate jobs in parallel using different GPUs/resources:

```bash
# Terminal 1: Joern CPG Generation
export JOERN_PATH=/shared/software/joern-cli
export WORKERS=32
bash scripts/generate_joern.sh

# Terminal 2: (after CPG is ready) Hetero processing
export WORKERS=64
export CUDA_DEVICE=0,1
bash scripts/generate_hetero.sh

# Terminal 3: (after CPG is ready) Homo processing
export WORKERS=64
export CUDA_DEVICE=2,3
bash scripts/generate_homo.sh
```

## Step 3: Environment Variable Reference

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `JOERN_PATH` | Path to joern-cli | None (required for Joern.py) | `/shared/software/joern-cli` |
| `WORKERS` | Number of parallel workers | 15 (Joern), 64 (others) | `32` |
| `WORKERS` | Parallel workers | 64 | `128` |
| `CUDA_DEVICE` | GPU devices to use | 1 (hetero), 0 (homo) | `0,1,2,3` |
| `SAVE` | Output directory | `CPG` | `/scratch/cpg_output` |
| `LIMIT` | Limit samples (testing) | None (full dataset) | `100` |

## Step 4: Monitor Progress

Check real-time logs:
```bash
# Monitor all running generations
watch -n 5 'le logs/*/Joern_generate_*.log; tail -20 logs/Hetero/*.log; tail -20 logs/Homo/*.log'

# Or individually
tail -f logs/Joern/Joern_generate_*.log
tail -f logs/Hetero/Hetero_generate_*.log
tail -f logs/Homo/Homo_generate_*.log
```

Check job status:
```bash
# Check if running
ps -p $(cat pids/Joern_generate_*.pid)
ps -p $(cat pids/Hetero_generate_*.pid)
ps -p $(cat pids/Homo_generate_*.pid)

# Or for SLURM
squeue -u $USER
```

## Step 5: Troubleshooting

### Issue: JOERN_PATH not found
**Solution:** Verify joern-cli path:
```bash
ls -la /shared/software/joern-cli/lib/ | wc -l  # Should show 100+
```

### Issue: Java not found
**Solution:** Set JAVA_HOME or load Java module:
```bash
export JAVA_HOME=/path/to/jdk
export PATH=$JAVA_HOME/bin:$PATH
java -version  # Verify
```

### Issue: Out of memory
**Solution:** Reduce workers:
```bash
export WORKERS=8  # Reduce from 64
bash scripts/generate_hetero.sh
```

### Issue: GPU out of memory
**Solution:** Set specific GPUs:
```bash
export CUDA_DEVICE=0  # Use only GPU 0
bash scripts/generate_hetero.sh
```

## Step 6: Output Structure

After successful completion:

```
CPG/
├── raw/
│   └── cpg_dataset.jsonl          # Input: Code + GraphML from Joern
├── processed_hetero/
│   ├── processed.pt               # Heterogeneous graph dataset
│   └── pre_transform.bin
└── processed_homo/
    ├── processed.pt               # Homogeneous graph dataset
    └── pre_transform.bin
```

## Tips for HPC

1. **Use node-local storage** for temporary files if available:
   ```bash
   export TMPDIR=/scratch/$SLURM_JOB_ID  # SLURM example
   ```

2. **Avoid main network storage** for temp Joern files:
   - Edit `utils/Joern.py` to use local temp: `temp_dir = "/lscratch/temp_joern"`

3. **Monitor disk space**:
   ```bash
   df -h  # Check space
   du -sh CPG/  # Check output size
   ```

4. **For large datasets**, increase cleanup frequency in `Joern.py`:
   - The script already cleans up temp files immediately (line 75-78)

## Performance Notes

- **Joern CPG generation**: ~5-15 samples/minute per worker (CPU-dependent)
- **Hetero/Homo processing**: ~100-500 samples/minute per worker
- **Total runtime for full dataset**: ~10-30 hours (depending on HPC specs)

For faster processing:
- Use more workers
- Allocate more CPU cores
- Use SSD-backed storage
- Run Joern and graph processing in parallel on different nodes
