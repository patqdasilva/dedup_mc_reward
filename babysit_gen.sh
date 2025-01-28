#!/bin/bash
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

trial=$1
if [ -z "$trial" ]; then
    echo "Error: Trial argument is missing."
    exit 1
fi
# Number of retries
MAX_RETRIES=10
RETRY_COUNT=0

# Path to the synthetic data
JSONL_FP="./synth_samples/gsm/"$trial".jsonl"
# Path to log
LOG_FP="./logs/data_gen/gsm_"$trial".log"

echo "[$TIMESTAMP] -------- running trial version ["$trial"] --------" >> "$LOG_FP"

# activate miniconda env
source ~/miniconda3/bin/activate dedup

# Get the length of the synthetic data file
# Use this to know the restart idx
get_jsonl_length() {
    LENGTH=$(python -c "import json; print(sum(1 for _ in open('$JSONL_FP')))")
    echo $LENGTH
}

# Rerun the script if it fails
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Free GPU memory
    if ! nvidia-smi | grep -q 'No running processes found'; then
        PIDS=$(nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//')
        for PID in $PIDS; do
            echo "Killing process $PID" >> "$LOG_FP"
            kill -9 $PID
        done
    fi
    # Get the length of the JSONL file
    JSONL_LENGTH=$(get_jsonl_length)
    echo "[$TIMESTAMP] JSONL file length: $JSONL_LENGTH" >> "$LOG_FP"

    python ./gen.py --trial $trial --restart_idx $JSONL_LENGTH
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$TIMESTAMP] Script ran successfully." >> "$LOG_FP"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "[$TIMESTAMP] Script failed with exit code $EXIT_CODE. Retrying ($RETRY_COUNT/$MAX_RETRIES)..." >> "$LOG_FP"
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "[$TIMESTAMP] Script failed after $MAX_RETRIES attempts." >> "$LOG_FP"
    exit 1
fi