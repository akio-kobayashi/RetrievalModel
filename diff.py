import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--set1", type=str, required=True)
parser.add_argument("--set2", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

df1 = pd.read_csv(args.set1)
df2 = pd.read_csv(args.set2)

diff = df1[~df1['key'].isin(df2['key'])]
diff.to_csv(args.out, index=False)
