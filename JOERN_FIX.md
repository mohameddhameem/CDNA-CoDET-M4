# Joern CPG Generation Fix & HPC Deployment

## Problem Summary
All 486,965 records in `CPG/raw/cpg_dataset.jsonl` had `graphml: null`, causing `cpg2hetero.py` to reject every record with the error:
```
Skipped 486965 invalid graphs.
No valid data found!
FileNotFoundError: [Errno 2] No such file or directory: 'CPG\\processed_hetero\\processed.pt'
```

## Root Cause
The `utils/Joern.py` script failed to generate GraphML because Joern CLI tools use shell wrapper scripts that have Windows path resolution issues. The exception was caught silently and `graphml: null` was written to the JSONL file.

## Solution Implemented

### 1. Fixed Joern.py (Direct Java Invocation)
**Before (shell wrapper approach):**
```python
self.joern_parse = os.path.join(joern_path, "joern-parse")
self.joern_export = os.path.join(joern_path, "joern-export")
```

**After (direct Java invocation):**
```python
lib_path = os.path.join(joern_path, "lib", "*")
self.joern_parse = ["java", "-cp", lib_path, "io.joern.joerncli.JoernParse"]
self.joern_export = ["java", "-cp", lib_path, "io.joern.joerncli.JoernExport"]
```

Also updated `parse_one()` method to handle list-based commands (lines 52, 59).

### 2. Added JOERN_PATH Configuration
**Added command-line argument:**
```bash
python utils/Joern.py --joern-path /path/to/joern-cli
```

**Or environment variable:**
```bash
export JOERN_PATH=/path/to/joern-cli
python utils/Joern.py
```

### 3. Updated All Generation Scripts
All three scripts now support environment variables for HPC deployment:
- `JOERN_PATH`: Path to joern-cli
- `WORKERS`: Number of parallel workers
- `CUDA_DEVICE`: GPU device selection
- `SAVE`: Output directory
- `LIMIT`: Sample limit (testing)

## Files Modified

| File | Changes |
|------|---------|
| `utils/Joern.py` | Use direct Java invocation, add --joern-path argument |
| `scripts/generate_joern.sh` | Add JOERN_PATH requirement, improve HPC compatibility |
| `scripts/generate_hetero.sh` | Add environment variable support |
| `scripts/generate_homo.sh` | Add environment variable support |
| `HPC_DEPLOYMENT.md` | **NEW** - Complete HPC setup guide |
| `HPC_QUICK_REF.md` | **NEW** - Quick reference for HPC users |

## Local Usage (Windows/Linux)

### Option 1: Using default path
```bash
python utils/Joern.py --workers 15 --path CPG
```

### Option 2: Using custom path
```bash
python utils/Joern.py --workers 15 --path CPG --joern-path C:\path\to\joern-cli
```

### Option 3: Using shell scripts
```bash
bash scripts/generate_joern.sh     # After renaming to use env var
bash scripts/generate_hetero.sh
bash scripts/generate_homo.sh
```

## HPC Usage

### Quick Start
```bash
# Load modules
module load java/19
module load python/3.11

# Set environment
export JOERN_PATH=/shared/software/joern-cli
export WORKERS=32

# Run pipeline
cd ~/projects/City-of-Agents-1
bash scripts/generate_joern.sh    # Generate CPGs
bash scripts/generate_hetero.sh   # Process to hetero
bash scripts/generate_homo.sh     # Process to homo
```

### Using SLURM
```bash
sbatch run_pipeline.sh
```

See `HPC_DEPLOYMENT.md` for complete HPC guide.

## Troubleshooting

### Issue: JOERN_PATH not found
```bash
# Verify joern-cli exists
ls -la /path/to/joern-cli/lib/ | wc -l  # Should show 100+
```

### Issue: Java not found
```bash
# Check Java
java -version

# Or set JAVA_HOME
export JAVA_HOME=/path/to/jdk
export PATH=$JAVA_HOME/bin:$PATH
```

### Issue: Still getting null graphml
```bash
# Check the error in subprocess
python -c "
from utils.Joern import JoernRunner
runner = JoernRunner(joern_path='./joern-cli')
result = runner.parse_one(0, 'x = 1')
print(result)
"
```

## Requirements

- **Java:** 8+ (tested: OpenJDK 19)
- **Python:** 3.8+
- **Joern:** CLI tools in `joern-cli/lib/` directory
- **Packages:** datasets, transformers, torch, torch-geometric, tqdm

## Workflow

```
1. utils/Joern.py           → CPG/raw/cpg_dataset.jsonl
                               (with valid GraphML)
                                     ↓
2. utils/cpg2hetero.py      → CPG/processed_hetero/
                                     ↓
3. utils/cpg2homo.py        → CPG/processed_homo/
```

## Performance Notes

- **Joern CPG generation:** ~5-15 samples/minute per worker
- **Graph processing:** ~100-500 samples/minute per worker
- **Full dataset runtime:** 10-30 hours (depending on cluster specs)

## References

- **Local Setup:** `JOERN_FIX.md` (this file)
- **HPC Deployment:** `HPC_DEPLOYMENT.md`
- **Quick Reference:** `HPC_QUICK_REF.md`

