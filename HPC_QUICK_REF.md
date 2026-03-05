# HPC Quick Reference Card

## Quick Start (Copy & Paste)

### 1. One-time Setup
```bash
# Copy to HPC
export JOERN_PATH=/shared/software/joern-cli
export JAVA_HOME=/shared/software/jdk-19
export PATH=$JAVA_HOME/bin:$PATH
export WORKERS=32

# Verify
java -version  # Should work
cd ~/projects/City-of-Agents-1
```

### 2. Run All Three Steps
```bash
# Step 1: Generate CPGs (longest, ~10-30 hours)
bash scripts/generate_joern.sh

# Step 2: Heterogeneous graphs (after step 1 completes)
bash scripts/generate_hetero.sh

# Step 3: Homogeneous graphs (after step 1 completes)
bash scripts/generate_homo.sh
```

### 3. Monitor Progress
```bash
# Watch logs
tail -f logs/Joern/Joern_generate_*.log
tail -f logs/Hetero/Hetero_generate_*.log
tail -f logs/Homo/Homo_generate_*.log
```

## Common Commands

### Run with custom workers
```bash
export WORKERS=64
bash scripts/generate_hetero.sh
```

### Run with specific GPU
```bash
export CUDA_DEVICE=0,1,2,3
bash scripts/generate_hetero.sh
```

### Test with small sample (10 samples)
```bash
export LIMIT=10
export WORKERS=1
bash scripts/generate_joern.sh
```

### Save to custom location
```bash
export SAVE=/scratch/my_cpg_data
bash scripts/generate_joern.sh
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `JOERN_PATH environment variable not set` | `export JOERN_PATH=/path/to/joern-cli` |
| `java: command not found` | `export JAVA_HOME=/path/to/jdk; export PATH=$JAVA_HOME/bin:$PATH` |
| `Error: Could not find or load main class` | Check: `ls -la $JOERN_PATH/lib/io.joern* \| wc -l` (should be >0) |
| `Memory error` | `export WORKERS=8` (reduce workers) |
| `Disk full` | Clean temp: `rm -rf CPG/temp_joern_workers` |

## Environment Variables Summary

```bash
# Set joern path (REQUIRED for Joern.py)
export JOERN_PATH=/path/to/joern-cli

# Set workers (optional, default: 15 for Joern, 64 for others)
export WORKERS=32

# Set GPU (optional, default: 1 for hetero, 0 for homo)
export CUDA_DEVICE=0

# Set output dir (optional, default: CPG)
export SAVE=CPG

# Test mode - limit samples (optional, default: all)
export LIMIT=100

# Then run scripts
bash scripts/generate_joern.sh
bash scripts/generate_hetero.sh
bash scripts/generate_homo.sh
```

## File Locations

```
Project Root: ~/projects/City-of-Agents-1/

Scripts:
  scripts/generate_joern.sh        # CPG generation via Joern
  scripts/generate_hetero.sh       # Heterogeneous graphs
  scripts/generate_homo.sh         # Homogeneous graphs

Code:
  utils/Joern.py                   # CPG generation code
  utils/cpg2hetero.py              # Hetero processing
  utils/cpg2homo.py                # Homo processing

Outputs:
  CPG/raw/cpg_dataset.jsonl        # Generated CPGs
  CPG/processed_hetero/            # Processed hetero data
  CPG/processed_homo/              # Processed homo data

Logs:
  logs/Joern/                      # Joern logs
  logs/Hetero/                     # Hetero logs
  logs/Homo/                       # Homo logs

PIDs (for monitoring):
  pids/Joern_generate_*.pid
  pids/Hetero_generate_*.pid
  pids/Homo_generate_*.pid
```

## Expected Output

```bash
# Successful run looks like:
# logs/Joern/Joern_generate_20250305_120000.log:
Processing: 100%|████████████████| 1000000/1000000 [2:15:30<00:00, 123.45it/s]
Done! All raw-code based CPGs saved to ./CPG/raw/cpg_dataset.jsonl

# logs/Hetero/Hetero_generate_20250305_130000.log:
Processing: 100%|████████████████| 1000000/1000000 [3:45:20<00:00, 74.32it/s]
Collating data...
Success! Saved to CPG/processed_hetero/processed.pt

# logs/Homo/Homo_generate_20250305_140000.log:
Processing: 100%|████████████████| 1000000/1000000 [2:15:40<00:00, 122.98it/s]
Collating data...
Success! Saved to CPG/processed_homo/processed.pt
```

## Tips

✅ **DO:**
- Run joern first, wait for completion before others
- Monitor logs: `tail -f logs/*/`*_generate_*.log
- Use `LIMIT=10` first to test setup
- Keep JOERN_PATH absolute (not relative)

❌ **DON'T:**
- Don't run all 3 scripts simultaneously (first time)
- Don't interrupt with Ctrl+C, use: `kill $(cat pids/*pid)`
- Don't use relative paths for JOERN_PATH
- Don't ignore errors - check logs first

## Full Pipeline Example

```bash
#!/bin/bash
cd ~/projects/City-of-Agents-1

# Setup
export JOERN_PATH=/shared/software/joern-cli
export JAVA_HOME=/shared/software/jdk-19
export PATH=$JAVA_HOME/bin:$PATH
export WORKERS=32

# Run with monitoring
echo "Step 1: Joern CPG Generation..."
bash scripts/generate_joern.sh
JOB1=$(cat pids/Joern_generate_*.pid | tail -1)
wait $JOB1
echo "Step 1 complete!"

echo "Step 2: Heterogeneous graphs..."
bash scripts/generate_hetero.sh
JOB2=$(cat pids/Hetero_generate_*.pid | tail -1)
wait $JOB2
echo "Step 2 complete!"

echo "Step 3: Homogeneous graphs..."
bash scripts/generate_homo.sh
JOB3=$(cat pids/Homo_generate_*.pid | tail -1)
wait $JOB3
echo "Pipeline complete!"

# Check output
echo "Final output sizes:"
du -sh CPG/raw/
du -sh CPG/processed_hetero/
du -sh CPG/processed_homo/
```
