#!/bin/bash

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

TASK_NAME="Hetero"
WORKERS=${WORKERS:-64}
COMMON_ARGS="--workers ${WORKERS}"

mkdir -p logs/${TASK_NAME}
mkdir -p pids

timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${TASK_NAME}/${TASK_NAME}_generate_${timestamp}.log
PID_FILE=pids/${TASK_NAME}_generate_${timestamp}.pid

echo "=== ${TASK_NAME} Generation Started ===" > $LOGFILE
echo "Workers: ${WORKERS}" >> $LOGFILE
: > $PID_FILE

nohup python utils/cpg2hetero.py ${COMMON_ARGS} \
    >> $LOGFILE 2>&1 &

echo $! >> $PID_FILE

echo "View real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if running: ps -p \$(cat $PID_FILE)" | tee -a $LOGFILE
echo "To stop the process: kill \$(cat $PID_FILE)" | tee -a $LOGFILE