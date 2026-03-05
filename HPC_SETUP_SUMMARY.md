# Summary: HPC-Ready Code Generation

## What Changed for HPC Compatibility

### Modified Files

#### 1. `utils/Joern.py`
**Changes:**
- Added `--joern-path` command-line argument
- Added fallback to `JOERN_PATH` environment variable
- Changed from shell wrapper scripts to direct Java invocation for better Windows/HPC compatibility

**Key lines:**
```python
# Line 21-23: Direct Java classpath invocation
lib_path = os.path.join(joern_path, "lib", "*")
self.joern_parse = ["java", "-cp", lib_path, "io.joern.joerncli.JoernParse"]
self.joern_export = ["java", "-cp", lib_path, "io.joern.joerncli.JoernExport"]

# Lines 52, 59: Handle both list and string command formats
parse_cmd = self.joern_parse if isinstance(self.joern_parse, list) else [self.joern_parse]
export_cmd = self.joern_export if isinstance(self.joern_export, list) else [self.joern_export]

# Lines 133-157: Add joern-path argument and env var support
parser.add_argument('--joern-path', type=str, default=None,
                   help='Path to joern-cli directory...')
joern_path = args.joern_path or os.environ.get('JOERN_PATH', r'C:\...')
```

#### 2. `scripts/generate_joern.sh` (Completely Rewritten)
**Changes:**
- Made JOERN_PATH a required environment variable
- Added comprehensive error checking
- Added better logging and status reporting
- HPC-friendly configuration

**Key features:**
```bash
# Validate JOERN_PATH is set
if [ -z "$JOERN_PATH" ]; then
    echo "ERROR: JOERN_PATH environment variable not set"
    exit 1
fi

# Pass JOERN_PATH to Python script
COMMON_ARGS="... --joern-path ${JOERN_PATH}"

# Enhanced logging with timestamp and configuration details
```

#### 3. `scripts/generate_hetero.sh` (Updated for HPC)
**Changes:**
- Made CUDA_DEVICE configurable via environment variable
- Unified logging format
- Better status reporting

**Key features:**
```bash
CUDA_DEVICE=${CUDA_DEVICE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
```

#### 4. `scripts/generate_homo.sh` (Updated for HPC)
**Changes:**
- Same improvements as generate_hetero.sh
- Configurable CUDA_DEVICE and WORKERS

### New Documentation Files

#### `HPC_DEPLOYMENT.md`
Complete guide covering:
- Prerequisites and setup
- 3 different deployment options (sequential, SLURM, parallel)
- Troubleshooting
- Performance notes
- Output structure explanation

#### `HPC_QUICK_REF.md`
Quick reference for HPC users:
- Copy-paste commands
- Common operations
- Troubleshooting table
- File locations
- Expected output examples

#### `JOERN_FIX.md` (Updated)
Comprehensive documentation of the fix including:
- Problem and root cause
- Solution details
- Local and HPC usage
- Troubleshooting guide

## How to Use on HPC

### Fast Track (Copy & Paste)

```bash
# 1. Set up once
export JOERN_PATH=/shared/software/joern-cli
export JAVA_HOME=/shared/software/jdk-19
export PATH=$JAVA_HOME/bin:$PATH
export WORKERS=32

# 2. Run the pipeline
cd ~/projects/City-of-Agents-1
bash scripts/generate_joern.sh    # ~10-30 hours
bash scripts/generate_hetero.sh   # ~5-10 hours
bash scripts/generate_homo.sh     # ~5-10 hours
```

### Key Differences from Local Setup

| Aspect | Local | HPC |
|--------|-------|-----|
| Path setup | Hardcoded or relative | Must use `JOERN_PATH` env var |
| Java | Expects in PATH | Must set `JAVA_HOME` |
| Workers | Default 15-64 | Adjust per cluster: 32-256+ |
| GPU | Use CUDA_DEVICE | Assign per job/node |
| Storage | Local disk | May need scratch space config |
| Execution | Direct or nohup | SLURM/PBS sbatch recommended |

## Deployment Checklist for HPC

- [ ] Copy `joern-cli/` to HPC's `/shared/software/` (or equivalent)
- [ ] Copy JDK to HPC's `/shared/software/` (or system Java available)
- [ ] Copy entire `City-of-Agents-1/` project to HPC home
- [ ] Verify Java: `java -version` returns 8+
- [ ] Verify Joern: `ls /shared/software/joern-cli/lib/ | wc -l` returns 100+
- [ ] Set `export JOERN_PATH=/shared/software/joern-cli`
- [ ] Test with small sample: `LIMIT=10 bash scripts/generate_joern.sh`
- [ ] Monitor with: `tail -f logs/Joern/Joern_generate_*.log`
- [ ] Once successful, run full pipeline

## Files to Transfer to HPC

```bash
# Minimum required for HPC
/shared/software/
├── joern-cli/           # Required - copy entire directory
└── jdk-19/              # Required - copy entire directory

~/projects/City-of-Agents-1/
├── utils/               # Required - Python code
├── scripts/             # Required - Generation scripts
├── CPG/                 # Required - Data directory
├── JOERN_FIX.md         # Recommended - Documentation
├── HPC_DEPLOYMENT.md    # Recommended - Setup guide
└── HPC_QUICK_REF.md     # Recommended - Quick commands
```

## Expected Execution Times (per 1M samples)

| Task | Workers | Time | Notes |
|------|---------|------|-------|
| Joern CPG | 8 | 200h | CPU intensive |
| Joern CPG | 32 | 50h | Recommended |
| Joern CPG | 64 | 25h | Fast (high CPU) |
| Hetero processing | 64 | 3-5h | GPU helpful |
| Homo processing | 64 | 2-4h | GPU helpful |

## Support & Debugging

If issues occur on HPC:

1. **Check environment:**
   ```bash
   echo $JOERN_PATH
   java -version
   python --version
   ```

2. **Check logs:**
   ```bash
   tail -100 logs/Joern/Joern_generate_*.log
   grep -i "error" logs/Joern/Joern_generate_*.log
   ```

3. **Test Joern directly:**
   ```bash
   java -cp "$JOERN_PATH/lib/*" io.joern.joerncli.JoernParse --help
   ```

4. **Check disk space:**
   ```bash
   df -h  # Check available space
   du -sh CPG/  # Check current usage
   ```

See `HPC_DEPLOYMENT.md` for detailed troubleshooting section.
