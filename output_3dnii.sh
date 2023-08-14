#!/bin/bash

DATA_DIR="/home/kszuyen/PIBNii_MPR_T1"
P_NAME="LowDose_with_T1"
TOTAL_FOLD=10
JSON_FILE="${TOTAL_FOLD}fold.json"
PROJECT_RESULTS_DIR="/home/kszuyen/MMIO_Low-Dose_PET_Enhancement/results/${P_NAME}"

for (( FOLD=1; FOLD<=$TOTAL_FOLD; FOLD++ ))
do
    python $(dirname "${BASH_SOURCE[0]}")/3dto2d_kfold.py -P $P_NAME --fold $FOLD --data_dir $DATA_DIR --total_fold $TOTAL_FOLD --json_file $JSON_FILE
done

python $(dirname "${BASH_SOURCE[0]}")/output_3dnii.py -P $P_NAME --json_file $JSON_FILE --project_results_dir $PROJECT_RESULTS_DIR
