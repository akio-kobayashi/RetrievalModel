from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys
import hubert_indexer as H

def main(args):
    H.index_from_tensor_directory(args.tensor_dir, args.memmap, args.index, args.metadata)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tensor_dir', type=str, required=True)
    parser.add_argument('--memmap', type=str, default='./memmap')
    parser.add_argument('--index', type=str, default='./index')
    parser.add_argument('--metadata', type=str, default='./metadata')

    args=parser.parse_args()
       
    main(args)
