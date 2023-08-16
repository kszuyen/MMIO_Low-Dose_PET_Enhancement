#!/bin/bash

P_NAME="LowDose"
TOTAL_FOLD=10
NUM_EPOCHS=800
LEARNING_RATE=0.00001
BATCH_SIZE=16
CUDA=0

for (( FOLD=1; FOLD<=$TOTAL_FOLD; FOLD++ ))
do
    for CASE in 1 2 3 4
    do
        python $(dirname "${BASH_SOURCE[0]}")/train.py -P $P_NAME --case $CASE --num_epochs $NUM_EPOCHS --fold $FOLD --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE --cuda $CUDA
    done
done