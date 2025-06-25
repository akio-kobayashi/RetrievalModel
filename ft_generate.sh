#!/bin/sh

ckpt='/media/akio/hdd1/rvc/models/adpt/M023/ckpt/epoch=38-val_loss=0.6874.ckpt'
stats=stats_mel.pt
csv=lora/M023_align_val.csv
mel_csv=lora/M023_mel_val.csv
out_dir='/media/akio/hdd1/rvc/models/adpt/M023/outputs/'
python3 ft_generate.py --ckpt $ckpt --stats $stats --csv $csv --mel_csv $mel_csv --out_dir $out_dir
