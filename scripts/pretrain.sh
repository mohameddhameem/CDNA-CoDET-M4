#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MODEL="full"
TASK="pretrain"
SEEDS=(0)
COMMON_ARGS="--model ${MODEL} \
    --task_name ${TASK} \
    --pattern pretrain \
    --path CPGs \
    --train_language all \
    --use_gpu True \
    --devices 0 \
    --gpu_type cuda \
    --batch_size 64 \
    --infer_batch_size 64 \
    --num_workers 4 \
    --learning_rate 0.0001 \
    --weight_decay 0.0 \
    --epochs 200 \
    --patience 5 \
    --dropout 0.1 \
    --input_dim 768 \
    --hidden_dim 768 \
    --output_dim 384 \
    --num_layers 3 \
    --alpha 1.0 \
    --beta 1.0 \
    --num_heads 0 \
    --temperature 1.0 \
    --is_logging True"

mkdir -p logs/${MODEL}
mkdir -p pids
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${MODEL}/${MODEL}_${TASK}_${timestamp}.log
PID_FILE=pids/${MODEL}_${TASK}_${timestamp}.pid

for SEED in "${SEEDS[@]}"
do
    echo "====================================" >> $LOGFILE
    echo "Seed: ${SEED}" >> $LOGFILE
    echo "====================================" >> $LOGFILE

    nohup python run.py ${COMMON_ARGS} --seed ${SEED} >> $LOGFILE 2>&1 &
    echo $! >> $PID_FILE
    sleep 5
done

echo "View real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE