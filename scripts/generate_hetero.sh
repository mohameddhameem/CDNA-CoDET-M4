#!/bin/bash

# ============================================================
# HPC-Friendly Heterogeneous Graph Generation Script
# ============================================================
# This script processes CPG data into heterogeneous graphs
#
# Usage:
#   bash scripts/generate_hetero.sh
#   WORKERS=64 SAVE=CPG bash scripts/generate_hetero.sh
#
# Optional environment variables:
#   WORKERS      - Number of parallel workers (default: 64)
#   CUDA_DEVICE  - GPU device ID for CUDA (default: 1)
#   SAVE         - Output directory (default: CPG)
# ============================================================

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Set defaults
CUDA_DEVICE=${CUDA_DEVICE:-1}
WORKERS=${WORKERS:-64}
SAVE=${SAVE:-CPG}

# Setup GPU
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

TASK_NAME="Hetero"
COMMON_ARGS="--workers ${WORKERS} --path ${SAVE}"

# Create directories
mkdir -p logs/${TASK_NAME}
mkdir -p pids

# Setup logging
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${TASK_NAME}/${TASK_NAME}_generate_${timestamp}.log
PID_FILE=pids/${TASK_NAME}_generate_${timestamp}.pid

{
    echo "=== ${TASK_NAME} Generation Started ==="
    echo "Timestamp: ${timestamp}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Workers: ${WORKERS}"
    echo "Save path: ${SAVE}"
    echo "Log: ${LOGFILE}"
    echo "========================================"
} | tee ${LOGFILE}

: > ${PID_FILE}

# Run in background
nohup python utils/cpg2hetero.py ${COMMON_ARGS} >> ${LOGFILE} 2>&1 &

echo $! >> ${PID_FILE}

echo ""
echo "Job submitted successfully!"
echo "View real-time logs with: tail -f $LOGFILE"
echo "Check status: ps -p \$(cat $PID_FILE)"
echo "Stop job: kill \$(cat $PID_FILE)"