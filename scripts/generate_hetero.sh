#!/bin/bash

# ============================================================
# HPC-Friendly Heterogeneous Graph Generation Script
# ============================================================
# This script processes CPG data into language-specific heterogeneous graphs
#
# Usage:
#   bash scripts/generate_hetero.sh                           # Processes python language
#   CODE_LANG=java WORKERS=64 bash scripts/generate_hetero.sh
#   CODE_LANG=cpp SAVE=CPG bash scripts/generate_hetero.sh
#
# Optional environment variables:
#   CODE_LANG    - Programming language to process (default: python)
#                 Options: python, java, cpp, or any language in dataset
#   WORKERS      - Number of parallel workers (default: 64)
#   CUDA_DEVICE  - GPU device ID for CUDA (default: 1)
#   SAVE         - Output directory (default: CPG)
#   NO_RESUME    - Set to 1 to disable resume and force full rebuild (default: 0)
#   CHECKPOINT_INTERVAL - Save resume checkpoint every N graphs (default: 500)
#   LIMIT        - Max records to process in this run (default: none = all rows)
# ============================================================

# Set JAVA_HOME to JDK-19 (required for CPG processing)
export JAVA_HOME=$HOME/software/jdk-19
export PATH=$JAVA_HOME/bin:$PATH

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Set defaults
CODE_LANG=${CODE_LANG:-python}
CUDA_DEVICE=${CUDA_DEVICE:-1}
WORKERS=${WORKERS:-64}
SAVE=${SAVE:-CPG}
NO_RESUME=${NO_RESUME:-0}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
LIMIT=${LIMIT:-}

# Setup GPU
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

TASK_NAME="Hetero_${CODE_LANG}"
COMMON_ARGS="--workers ${WORKERS} --path ${SAVE} --language ${CODE_LANG} --checkpoint-interval ${CHECKPOINT_INTERVAL}"
if [ -n "${LIMIT}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --limit ${LIMIT}"
    LIMIT_DISPLAY="${LIMIT}"
else
    LIMIT_DISPLAY="none (all rows)"
fi
if [ "${NO_RESUME}" = "1" ]; then
    COMMON_ARGS="${COMMON_ARGS} --no-resume"
    RESUME_MODE="disabled"
else
    RESUME_MODE="enabled"
fi

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
    echo "Language: ${CODE_LANG}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Workers: ${WORKERS}"
    echo "Save path: ${SAVE}"
    echo "Resume mode: ${RESUME_MODE}"
    echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}"
    echo "Limit: ${LIMIT_DISPLAY}"
    echo "Expected input: ${SAVE}/raw/cpg_dataset_${CODE_LANG}.jsonl"
    echo "Output: ${SAVE}/processed_hetero_${CODE_LANG}/"
    if [ "${RESUME_MODE}" = "enabled" ] && [ -f "${SAVE}/processed_hetero_${CODE_LANG}/processed.pt" ]; then
        echo "Note: Existing processed.pt detected; run may load and exit quickly. Set NO_RESUME=1 for full rebuild."
    fi
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