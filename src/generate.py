## Generate tokens using the trained neural language model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

from time import time
from datetime import datetime

from dataUtils import Data, Sample
from model import langModel
from utils import Name
from pytorchUtils import save_model, load_model, accumulate_grads, load_grads, save_grads

import numpy as np
import itertools
import json
import pdb
from tqdm import tqdm
import sys
import os

import argparse

def loop(args, exp):
  assert args.load, 'Model name not provided'
  assert os.path.isfile(args.load), 'Model file not found'

  ## load args from the saved file if it exists
  args_filepath = '_'.join(args.load.split('_')[:-1] + ['args.args'])
  if os.path.isfile(args_filepath):
    args_dict = json.load(open(args_filepath))
    args_dict.update({'load':args.load})
    args.__dict__.update(args_dict)
  
  ## Training parameters
  delimiter = args.delimiter
  file_path = args.data
  num_unrolling = args.num_unrolling
  num_layers = args.num_layers
  lru_layers = args.lru_layers
  rnn_size = args.rnn_size
  encoding = args.encoding
  num_sample = args.num_sample

  ## Load data iterables
  train = Data(file_path, batch_size=1, num_unrolling=num_unrolling, delimiter=delimiter, suffix='.train', encoding=args.encoding)
  vocab_size = len(train.vocab)

  ## Create Model
  model = langModel(vocab_size=vocab_size, hidden_size=rnn_size, num_layers=num_layers, lru_layers=lru_layers, model=args.model)
  if args.cuda >=0:
    model.cuda(args.cuda)

  ## Load model
  print('Loading Model')
  load_model(model, args.load)
  def generate(model, num_sample, data):
    mem = None

    ## choose a random starting point from the vocabulary
    index = Sample.sample_distribution(np.ones(vocab_size, dtype=np.float)/vocab_size)
    output = data.id2token(index)
    in_value = np.zeros((1, 1))
    in_value[0,0] = index
    for i in range(num_sample):
      x = Variable(torch.LongTensor(np.asarray(in_value, dtype=np.int)))
      if args.cuda>=0:
        x = x.cuda(args.cuda)

      model.eval()
      x_cap, mem = model(x, mem)
      index = Sample.sample_distribution(F.softmax(x_cap).data.cpu().numpy().squeeze())
      output += data.id2token(index)
      in_value[0,0] = index

      # detaching all variables from the model    
      x.detach_()


    return output

  output = generate(model, num_sample, train)
  print(output.encode(encoding=encoding))
  
if __name__=='__main__':
  parser = argparse.ArgumentParser()

  ## I/O for the model 
  parser.add_argument('-data', nargs='+', type=str, default=['../dataset/ptb/ptb.all.txt'],
                      help='path to data')
  parser.add_argument('-delimiter', nargs='+', type=str, default=[''],
                      help='kind of delimiter to tokenize data')
  parser.add_argument('-encoding', nargs='+', type=str, default=['utf-8'],
                      help='encoding of the data used; utf-8 or cp037')
  parser.add_argument('-num_sample', nargs='+', type=int, default=[1000],
                      help='Number of sample tokens to be printed')
  parser.add_argument('-save_dir', nargs='+', type=str, default=['save/model'],
                      help='directory to store checkpointed models')
  parser.add_argument('-cpk', nargs='+', type=str, default=['m'],
                      help='checkpointed model name')
  parser.add_argument('-load', nargs='+', type=str, default=[None],
                      help='Load weights from this file')
  parser.add_argument('-cuda', nargs='+', type=int, default=[0],
                      help='choice of gpu device, -1 for cpu')

  ## model hyperparameters
  parser.add_argument('-model', nargs='+', type=str, default=['lru'],
                      help='choice of model to train on [lru, gru, lstm, glstm, highway, lru2, lru3]')
  parser.add_argument('-rnn_size', nargs='+', type=int , default=[512],
                      help='size of RNN hidden state') 
  parser.add_argument('-num_layers', nargs='+', type=int, default=[2],
                      help='number of layers in the RNN')
  parser.add_argument('-num_unrolling', nargs='+', type=int, default=[50],
                      help='number of unrollings in time')
  parser.add_argument('-lru_layers', nargs='+', type=int, default=[1],
                      help='number of layers in the lru unit')

  args, unknown = parser.parse_known_args()
  print(args)
  print(unknown)

  ## Create a permutation of all the values in argparse
  args_dict = args.__dict__
  args_keys = sorted(args_dict)
  args_perm = [dict(zip(args_keys, prod)) for prod in itertools.product(*(args_dict[names] for names in args_keys))]

  for i, perm in enumerate(args_perm):
    args.__dict__.update(perm)
    print(args)
    loop(args, i)
