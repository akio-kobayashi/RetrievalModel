#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/akio/ResidualModel:$PYTHONPATH"

root=/media/akio/hdd1/
dir_header=${root}/limited_deaf_cloning_
output_dir=/media/akio/hdd1/retrieval/deaf/
mkdir ${output_dir}
for name in A B C D E F;
do
    python3 extract_feature_from_dir.py --dir ${dir_header}${name} --output_dir ${output_dir}
done

dir_header=${root}/limited_deaf_cloning_fake_
output_dir=/media/akio/hdd1/retrieval/fake/
mkdir ${output_dir}
for name in A B C D E F;
do
    python3 extract_feature_from_dir.py --dir ${dir_header}${name} --output_dir ${output_dir}
done
