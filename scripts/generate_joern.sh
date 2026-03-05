#!/bin/bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

TASK_NAME="Joern"
WORKERS=${WORKERS:-15}
LIMIT=${LIMIT:-}  # Optional limit for testing
SAVE="CPG"

COMMON_ARGS="--workers ${WORKERS} --path ${SAVE}"

if [ -n "$LIMIT" ]; then
    COMMON_ARGS="$COMMON_ARGS --limit $LIMIT"
fi

mkdir -p logs/${TASK_NAME}
mkdir -p pids

timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${TASK_NAME}/${TASK_NAME}_generate_${timestamp}.log
PID_FILE=pids/${TASK_NAME}_generate_${timestamp}.pid

echo "=== ${TASK_NAME} Generation Started ===" > $LOGFILE
echo "Workers: ${WORKERS}" >> $LOGFILE
if [ -n "$LIMIT" ]; then
    echo "Limit: ${LIMIT}" >> $LOGFILE
fi
: > $PID_FILE

nohup python utils/Joern.py ${COMMON_ARGS} \
    >> $LOGFILE 2>&1 &

echo $! >> $PID_FILE

echo "View real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if running: ps -p \$(cat $PID_FILE)" | tee -a $LOGFILE
echo "To stop the process: kill \$(cat $PID_FILE)" | tee -a $LOGFILE
