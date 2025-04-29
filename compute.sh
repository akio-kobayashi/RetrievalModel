#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/akio/ResidualModel:$PYTHONPATH"

output_dir=/media/akio/hdd1/residual/data/
mkdir -p ${output_dir}
python3 extract_feature.py --wave_csv jnas_orig.csv --output_dir ${output_dir}
