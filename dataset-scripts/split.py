'''
Create smaller text files of the given text file by taking some percentage off the start of the file.

Usage:
 ` python split.py -data path/to/data -encoding utf-8 -splits 5 10 `
'''

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default='../dataset/ptb/ptb.all.txt',
                    help='path to data')
parser.add_argument('-encoding', type=str, default='utf-8',
                    help='encoding of the data used; utf-8 or cp037')
parser.add_argument('-splits', nargs='+', type=str, default=[5,10,30,50,70,90],
                    help='create splits of args.splits percent')

args, unknown = parser.parse_known_args()
print(args)
print(unknown)

with open(args.data, encoding=args.encoding) as f:
  data = f.read()
  data_len = len(data)
  for num in args.splits:
    num = int(num)
    with open(args.data+'.%d'%(num), 'w', encoding=args.encoding) as g:
      g.write(data[:int(data_len*num/100)])
