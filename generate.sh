#!/bin/sh
python3 generate.py --csv test.csv --ckpt /media/akio/hdd1/rvc/models/stft/logs/ckpt/last.ckpt \
	--out_dir /media/akio/hdd1/rvc/models/stft/logs/outs --stats stats_pitch.pt
