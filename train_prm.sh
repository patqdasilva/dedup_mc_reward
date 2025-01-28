#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

trial=$1
if [ -z "$trial" ]; then
    echo "Error: Trial argument is missing."
    exit 1
fi
n_steps=$2
if [ -z "$n_steps" ]; then
    echo "Error: n_eval_steps argument is missing."
    exit 1
fi

LOG_FP="./logs/training/outlogs/gsm_"$trial".log"

echo "[$TIMESTAMP] -------- running trial version ["$trial"] --------" >> "$LOG_FP"

# Free GPU memory
if ! nvidia-smi | grep -q 'No running processes found'; then
    PIDS=$(nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//')
    for PID in $PIDS; do
        echo "Killing process $PID" >> "$LOG_FP"
        kill -9 $PID
    done
fi
# activate miniconda env
source ~/miniconda3/bin/activate nlp

python ./train_prm.py --trial $trial --n_eval_steps $n_steps >> "$LOG_FP"

