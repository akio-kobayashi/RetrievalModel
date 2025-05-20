#!/bin/sh
python3 generate.py --csv test.csv --ckpt /media/akio/hdd1/rvc/models/test/logs/ckpt/last.ckpt \
	--out_dir /media/akio/hdd1/rvc/models/test/logs/outs --stats stats_pitch.pt
