from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys
import torch

def main(args):

  keys, srcs, tgts = [], [], []
  source_files = {}
  for idx, filepath in enumerate(sorted(glob.glob(os.path.join(args.source, '*.pt'))), start=1):
    key = os.path.splitext(os.path.basename(filepath))[0].replace('_fake', '')
    source_files[key] = filepath

  for idx, filepath in enumerate(sorted(glob.glob(os.path.join(args.target, '*.pt'))), start=1):
    key = os.path.splitext(os.path.basename(filepath))[0].replace('_fake', '')
    if key in source_files.keys():
      keys.append(key)
      srcs.append(source_files[key])
      tgts.append(filepath)

  result = pd.DataFrame.from_dict({'key': keys, 'source': srcs, 'target': tgts })
  result.to_csv(args.output_csv, index=False)

  for idx, row in result.iterrows():
    src_dict = torch.load(row['source'], map_location='cpu')
    src_hubert = src_dict['hubert'].float()   # (T_src, 768)
    tgt_dict = torch.load(row['target'], map_location='cpu')
    tgt_hubert = tgt_dict['hubert'].float()   # (T_src, 768)
    print(src_hubert.shape, tgt_hubert.shape)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args=parser.parse_args()
       
    main(args)