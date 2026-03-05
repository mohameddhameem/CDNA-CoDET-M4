#!/bin/bash

# ============================================================
# HPC-Friendly CPG Generation Script
# ============================================================
# This script generates CPG dataset with GraphML via Joern
#
# Usage:
#   bash scripts/generate_joern.sh
#   WORKERS=32 LIMIT=100 bash scripts/generate_joern.sh
#   JOERN_PATH=/custom/path/to/joern-cli bash scripts/generate_joern.sh
#
# Optional environment variables:
#   JOERN_PATH   - Path to joern-cli directory (default: /storage/home/dhameem.m.2025/bin/joern/joern-cli)
#   WORKERS      - Number of parallel workers (default: 15)
#   LIMIT        - Limit number of samples for testing (default: none)
#   SAVE         - Output directory (default: CPG)
# ============================================================

# Set defaults
WORKERS=${WORKERS:-15}
LIMIT=${LIMIT:-2}
SAVE=${SAVE:-CPG}
JOERN_PATH=${JOERN_PATH:-/storage/home/dhameem.m.2025/bin/joern/joern-cli}

# Check JOERN_PATH exists
if [ ! -d "$JOERN_PATH" ]; then
    echo "ERROR: JOERN_PATH does not exist: $JOERN_PATH"
    exit 1
fi

# Verify joern lib directory exists
if [ ! -d "$JOERN_PATH/lib" ]; then
    echo "ERROR: Joern lib directory not found at: $JOERN_PATH/lib"
    exit 1
fi

# Setup Python environment
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

TASK_NAME="Joern"

# Create directories if they don't exist
mkdir -p logs/${TASK_NAME}
mkdir -p pids

# Setup logging
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${TASK_NAME}/${TASK_NAME}_generate_${timestamp}.log
PID_FILE=pids/${TASK_NAME}_generate_${timestamp}.pid

# Log configuration
{
    echo "=== ${TASK_NAME} Generation Started ==="
    echo "Timestamp: ${timestamp}"
    echo "JOERN_PATH: ${JOERN_PATH}"
    echo "Workers: ${WORKERS}"
    echo "Save path: ${SAVE}"
    [ -n "$LIMIT" ] && echo "Limit: ${LIMIT}" || echo "Limit: none (full dataset)"
    echo "Log: ${LOGFILE}"
    echo "PID: ${PID_FILE}"
    echo "========================================"
} | tee ${LOGFILE}

: > ${PID_FILE}

# Build arguments
COMMON_ARGS="--workers ${WORKERS} --path ${SAVE} --joern-path ${JOERN_PATH}"
[ -n "$LIMIT" ] && COMMON_ARGS="$COMMON_ARGS --limit $LIMIT"

# Run in background
nohup python utils/Joern.py ${COMMON_ARGS} >> ${LOGFILE} 2>&1 &

echo $! >> ${PID_FILE}

echo ""
echo "Job submitted successfully!"
echo "View real-time logs with: tail -f $LOGFILE"
echo "Check status: ps -p \$(cat $PID_FILE)"
echo "Stop job: kill \$(cat $PID_FILE)"
