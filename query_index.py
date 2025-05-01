from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys
from hubert_retriever import IVFPQFaissRetriever
import hubert_retriever as H
import numpy as np
import torch
from einops import rearrange

def main(args):
    #df = pd.read_csv(args.csv)
    retr = IVFPQFaissRetriever(
        index_path=args.index,
        memmap_path=args.memmap,
        metadata_path=args.metadata,
    )
    
    #for idx, row in df.iterrows():
    #q = H.load_query_tensor(row['source'])
    q = H.load_query_tensor(args.file)
    print(q.shape)
    vecs, infos = retr.search(q, topk=args.topk)  # unpack single query
    vecs = rearrange(torch.from_numpy(vecs), 't b f -> b t f')
    print(vecs.shape)
    #dst_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(row['source']))[0] + '.pt')
    #H.replace_hubert_in_tensor(row['source'], vecs, dst_path)
    dst_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.file))[0] + '.pt')
    H.replace_hubert_in_tensor(args.file, vecs.cpu(), dst_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--file', type=str )
    parser.add_argument('--memmap', type=str, default='./memmap')
    parser.add_argument('--index', type=str, default='./index')
    parser.add_argument('--metadata', type=str, default='./metadata')
    parser.add_argument('--topk', type=int, default=1)


    args=parser.parse_args()
       
    main(args)
