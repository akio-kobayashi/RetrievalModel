#!/bin/sh

ckpt=
stats=
csv=
mel_csv=
out_dir=
python3 -s ft_generate.py --ckpt $ckpt --stats $stats --csv $csv --mel_csv $csv --out_dir $out_dir
