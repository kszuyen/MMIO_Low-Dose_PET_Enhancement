#!/bin/bash

P_NAME="LowDose"

python $(dirname "${BASH_SOURCE[0]}")/output_3dnii.py -P $P_NAME
python $(dirname "${BASH_SOURCE[0]}")/output_2dpng.py -P $P_NAME
python $(dirname "${BASH_SOURCE[0]}")/calculate_eval_metric.py -P $P_NAME
