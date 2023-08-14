#!/bin/bash

DATA_DIR="/home/kszuyen/PIBNii_MPR_T1"
P_NAME="LowDose"
TOTAL_FOLD=10
JSON_FILE="${TOTAL_FOLD}fold.json"
PROJECT_RESULTS_DIR="./results/${P_NAME}"

for (( FOLD=1; FOLD<=$TOTAL_FOLD; FOLD++ ))
do
    python $(dirname "${BASH_SOURCE[0]}")/3dto2d_kfold.py -P $P_NAME --fold $FOLD --data_dir $DATA_DIR --total_fold $TOTAL_FOLD --json_file $JSON_FILE
done

python $(dirname "${BASH_SOURCE[0]}")/calculate_eval_metric.py -P $P_NAME --json_file $JSON_FILE --project_results_dir $PROJECT_RESULTS_DIR