#!/bin/bash

DATA_DIR="/home/kszuyen/PIBNii_MPR_T1"
P_NAME="LowDose_with_T1"
TOTAL_FOLD=10
JSON_FILE="${TOTAL_FOLD}fold.json"
NUM_EPOCHS=800
LEARNING_RATE=0.00001
CUDA=0

python $(dirname "${BASH_SOURCE[0]}")/split.py --data_dir $DATA_DIR --total_fold $TOTAL_FOLD --json_file $JSON_FILE
for (( FOLD=1; FOLD<=$TOTAL_FOLD; FOLD++ ))
do
    python $(dirname "${BASH_SOURCE[0]}")/3dto2d_kfold.py -P $P_NAME --fold $FOLD --data_dir $DATA_DIR --total_fold $TOTAL_FOLD --json_file $JSON_FILE
    for CASE in 1 2 3 4
    do
        python $(dirname "${BASH_SOURCE[0]}")/train.py -P $P_NAME --case $CASE --num_epochs $NUM_EPOCHS --fold $FOLD --learning_rate $LEARNING_RATE --cuda $CUDA
    done
    # rm -rf $(dirname "${BASH_SOURCE[0]}")/2d_data_"${P_NAME}"_fold"${FOLD}"/
done