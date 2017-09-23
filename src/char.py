import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

from time import time
from datetime import datetime

from dataUtils import Data
from model import langModel
from utils import Name
from pytorchUtils import save_model, load_model, accumulate_grads, load_grads, save_grads

import numpy as np
import itertools
import json
import pdb
from tqdm import tqdm
import sys
import copy

import argparse

def train(args, exp_num):
  # Name class to decide the filenames
  name = Name(args, 'cpk', 'model', 'rnn_size', 'num_layers', 'lru_layers', 'num_unrolling', 'num_epochs')

  # Start Log
  with open(name('log','log', args.save_dir), 'w') as f:
    f.write("S: %s\n"%(str(datetime.now())))
  
  ## Training parameters
  file_path = args.data
  batch_size = args.batch_size
  num_unrolling = args.num_unrolling
  delimiter = args.delimiter
  num_layers = args.num_layers
  lru_layers = args.lru_layers
  rnn_size = args.rnn_size
  num_epochs = args.num_epochs
  best_val_score = np.inf
  
  ## Load data iterables
  train = Data(file_path, batch_size=batch_size, num_unrolling=num_unrolling, delimiter=delimiter, suffix='.train', encoding=args.encoding, valid_frac=args.valid_frac, test_frac=args.test_frac)
  val = Data(file_path, batch_size=batch_size, num_unrolling=num_unrolling, delimiter=delimiter, suffix='.val', vocab=train.vocab, rev_vocab=train.rev_vocab, encoding=args.encoding, valid_frac=args.valid_frac, test_frac=args.test_frac)
  test = Data(file_path, batch_size=batch_size, num_unrolling=num_unrolling, delimiter=delimiter, suffix='.test', vocab=train.vocab, rev_vocab=train.rev_vocab, encoding=args.encoding, valid_frac=args.valid_frac, test_frac=args.test_frac)
  vocab_size = len(train.rev_vocab)

  ## Create a model
  model = langModel(vocab_size=vocab_size, hidden_size=rnn_size, num_layers=num_layers, lru_layers=lru_layers, model=args.model)
  if args.cuda >=0:
    model.cuda(args.cuda)
  best_model = copy.deepcopy(model).cpu()

  ## Load model
  if args.load:
    print('Loading Model')
    load_model(model, args.load)

  ## Loss function
  def criterion(x_cap, y):
    x_cap = torch.log(F.softmax(x_cap))/torch.log(Variable(x_cap.data.new(1).zero_()+2))
    return F.nll_loss(x_cap, y)
  
  criterion = torch.nn.CrossEntropyLoss()

  ## Optimizers
  decay_rate = args.decay_rate
  optim = torch.optim.Adam(model.parameters())
  def decayLR(optim, epoch, decay_rate, init_lr=0.001, lr_decay_epoch=1):
    lr = init_lr * (decay_rate**(epoch // lr_decay_epoch))
    for param_group in optim.param_groups:
      param_group['lr'] = lr
    tqdm.write('lr:%f'%(lr))
    return optim
  
  def validation(model, data, desc):
    running_loss = 0
    mem = None
    for count, batch in tqdm(enumerate(data), desc=desc, leave=False):
      x = Variable(torch.LongTensor(np.asarray(batch[:,:-1], dtype=np.int)))
      y = Variable(torch.LongTensor(np.asarray(batch[:, 1:], dtype=np.int)))
      if args.cuda>=0:
        x = x.cuda(args.cuda)
        y = y.cuda(args.cuda)
      y = y.view(-1)

      model.zero_grad()
      model.eval()
      x_cap, mem = model(x, mem)
      loss = criterion(x_cap, y)
      running_loss += loss.data[0]

      # detaching all variables from the model    
      x.detach_()
      y.detach_()
      loss.detach_()
      x_cap.detach_()

    return running_loss/(count+1.)

  ## Results
  if args.load:
    print('Loading results')
    res = json.load(open('_'.join(args.load.split('.')[0].split('_')[:-1]) +'_res.json'))
  else:
    res = {'train':[], 'val':[], 'test':[]}

  stop_count = 0
  stop_thresh = args.stop_thresh
  ## Training Loop
  for epoch in tqdm(range(num_epochs)):
    running_loss = 0
    grads_list = []
    mem = None

    ## learning rate decay
    optim = decayLR(optim, epoch, decay_rate)

    for count, batch in tqdm(enumerate(train), desc='train'):
      x = Variable(torch.LongTensor(np.asarray(batch[:,:-1], dtype=np.int)))
      y = Variable(torch.LongTensor(np.asarray(batch[:, 1:], dtype=np.int)))
      if args.cuda>=0:
        x = x.cuda(args.cuda)
        y = y.cuda(args.cuda)
      y = y.view(-1)

      model.zero_grad()
      optim.zero_grad()
      model.train()
      x_cap, mem = model(x, mem)
      loss = criterion(x_cap, y)
      running_loss += loss.data[0]
      loss.backward()
      if args.save_grads:
        grads_list = accumulate_grads(model.rnn, grads_list)
        bc = [i.grad.data for i in model.rnn.parameters()]
        ab = grads_list[0].clone()
      optim.step()

      # detaching all variables from the model
      x.detach_()
      y.detach_()
      loss.detach_()
      x_cap.detach_()
    val_loss = validation(model, val, 'val')
    test_loss = validation(model, test, 'test')

    ## save results
    res['train'].append(running_loss/(count+1.))
    res['val'].append(val_loss)
    res['test'].append(test_loss)
    json.dump(res, open(name('res', 'json', args.save_dir),'w'))

    tqdm.write("Exp:%d, Epch: %d, Train: %f(%f), Val: %f(%f), Test: %f(%f)"%(exp_num, epoch, res['train'][-1], np.exp(res['train'][-1]), res['val'][-1], np.exp(res['val'][-1]), res['test'][-1], np.exp(res['test'][-1])))


    ## save the model
    if res['val'][-1]<best_val_score:
      if args.greedy_save:
        save_flag = True
      else:
        best_model = copy.deepcopy(model).cpu()
        save_flag=False
      best_val_score = res['val'][-1]
    else:
      save_flag = False

    ## debug mode with no saving
    if not args.save_model:
      save_flag = False

    if save_flag:
      tqdm.write('Saving Model Greedily')
      save_model(model, name('weights','p',args.save_dir))

    ## Save grads
    if args.save_grads:
      # for the first epoch create a new file
      tqdm.write('Saving Grads')
      grads_list_name = name('grads','p', args.save_dir)
      if epoch == 0:
        save_grads([grads_list], grads_list_name)
      else:
        cummulative_grads_list = load_grads(grads_list_name)
        cummulative_grads_list.append(grads_list)
        save_grads(cummulative_grads_list, grads_list_name)
          
    ## early_stopping
    if args.early_stopping and len(res['train'])>=2:
      if (res['val'][-2] - args.eps < res['val'][-1]):
        stop_count += 1
      else:
        stop_count = 0

    if stop_count >= stop_thresh:
      print('Validation Loss is increasing')
      ## save the best model now
      if args.save_model and not args.greedy_save:
        print('Saving Model at the end of training')
        save_model(best_model, name('weights','p',args.save_dir))
      break

    ## end of training loop
    if epoch == num_epochs-1:
      print('Saving model after exceeding number of epochs')
      save_model(best_model, name('weights','p',args.save_dir))
      
  # End Log
  with open(name('log','log', args.save_dir), 'a') as f:
    f.write("E: %s"%(str(datetime.now())))

      
if __name__ == '__main__':
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
                      help='choice of model to train on [lru, gru, lstm, glstm, highway, rglru, pslru]')
  parser.add_argument('-rnn_size', nargs='+', type=int , default=[512],
                      help='size of RNN hidden state') 
  parser.add_argument('-num_layers', nargs='+', type=int, default=[2],
                      help='number of layers in the RNN')
  parser.add_argument('-num_unrolling', nargs='+', type=int, default=[50],
                      help='number of unrollings in time')
  parser.add_argument('-lru_layers', nargs='+', type=int, default=[1],
                      help='number of layers in the lru unit')

  ## training parameters
  parser.add_argument('-num_epochs', nargs='+', type=int, default=[20],
                      help='number of epochs for training')
  parser.add_argument('-early_stopping', nargs='+', type=int, default=[1],
                      help='Use 1 for early stopping')
  parser.add_argument('-stop_thresh', nargs='+', type=int, default=[3],
                      help='number of consequetive validation loss increses before stopping')
  parser.add_argument('-eps', nargs='+', type=float, default=[0.0001],
                      help='if the decrease in validation is less than eps, it counts for one step in stop_thresh ')
  parser.add_argument('-greedy_save', nargs='+', type=int, default=[0],
                      help='save weights after each epoch if 1')
  parser.add_argument('-save_grads', nargs='+', type=int, default=[0],
                      help='save weights after each epoch if 1')
  parser.add_argument('-save_model', nargs='+', type=int, default=[1],
                      help='if false the model will not be saved')
  parser.add_argument('-batch_size', nargs='+', type=int, default=[100],
                      help='minibatch size')
  parser.add_argument('-valid_frac', nargs='+', type=float, default=[0.05],
                      help='Fraction of data to be used as validation')
  parser.add_argument('-test_frac', nargs='+', type=float, default=[0.05],
                      help='Fraction of data to be used as test')

  ## optimization paramters
  parser.add_argument('-lr', nargs='+', type=float, default=[0.001],
                      help='learning rate')
  parser.add_argument('-decay_rate', nargs='+', type=float, default=[1],
                       help='decay rate for the learning rate')

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
    train(args, i)
