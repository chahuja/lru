import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
import pdb
from time import time
import math

class RGLRUCell(nn.Module):
  '''
  Decoupled Projected state and Reset Gate

  LRU is 2-dimensional, hence it has 2 inputs and 2 outputs
  
  Args:
    hidden_size: The number of features in the hidden state h (also called the size of LRU),
                also number of features of the input (both dimensions have the same input)
    bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True

  Inputs: (h1,h2) or (h1) or h1
    - **h1** (batch, hidden_size): tensor containing h1 features
    - **h2** (batch, hidden_size): (optional; will be set to zero if not provided) 
                                   tensor containing h2 features

  Outputs: h1, h2
    - **h1** (batch, hidden_size): tensor containing the next h1
    - **h2** (batch, hidden_size): tensor containing the next h2

  '''
  def __init__(self, hidden_size, bias=True):
    super(RGLRUCell, self).__init__()
    self.lin = nn.Linear(hidden_size*2, hidden_size*3, bias=bias)
    self.gate0 = nn.Linear(hidden_size*2, hidden_size)
    self.gate1 = nn.Linear(hidden_size*2, hidden_size)

    self.reset_parameters([self.lin, self.gate0, self.gate1], hidden_size)
    
  def reset_parameters(self, module_list, hidden_size):
    ## glorot initialization
    stdv = 1. / math.sqrt(hidden_size)
    for module in module_list:
      for i, param in enumerate(module.parameters()):
        param.data.uniform_(-stdv, stdv)
    
  def forward(self, inputs):
    if type(inputs) is not tuple:
      inputs = tuple([inputs])
    if len(inputs) == 1:
      inputs = tuple([inputs[0], Variable(inputs[0].data.new(*inputs[0].size()).zero_())])
      
    inputs_cat = torch.cat(inputs, dim=1)
    g = F.sigmoid(self.lin(inputs_cat))
    z, r, q = torch.chunk(g, chunks=3, dim=1)
    h0_cap = torch.cat([inputs[0], inputs[1]*r], dim=1)
    h1_cap = torch.cat([inputs[1], inputs[0]*q], dim=1)                   
    h0_cap = F.tanh(self.gate0(h0_cap))
    h1_cap = F.tanh(self.gate1(h1_cap))

    return z*h1_cap + (1.-z)*inputs[0], z*h0_cap + (1.-z)*inputs[1]

class LRUCell(nn.Module):
  '''
  Decoupled Projected state, Reset Gate and Output Gate

  LRU is 2-dimensional, hence it has 2 inputs and 2 outputs
  
  Args:
    hidden_size: The number of features in the hidden state h (also called the size of LRU),
                also number of features of the input (both dimensions have the same input)
    bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True

  Inputs: (h1,h2) or (h1) or h1
    - **h1** (batch, hidden_size): tensor containing h1 features
    - **h2** (batch, hidden_size): (optional; will be set to zero if not provided) 
                                   tensor containing h2 features

  Outputs: h1, h2
    - **h1** (batch, hidden_size): tensor containing the next h1
    - **h2** (batch, hidden_size): tensor containing the next h2

  '''
  def __init__(self, hidden_size, bias=True):
    super(LRUCell, self).__init__()
    self.lin = nn.Linear(hidden_size*2, hidden_size*4, bias=bias)
    self.gate0 = nn.Linear(hidden_size*2, hidden_size)
    self.gate1 = nn.Linear(hidden_size*2, hidden_size)

    self.reset_parameters([self.lin, self.gate0, self.gate1], hidden_size)
    
  def reset_parameters(self, module_list, hidden_size):
    ## glorot initialization
    stdv = 1. / math.sqrt(hidden_size)
    for module in module_list:
      for i, param in enumerate(module.parameters()):
        param.data.uniform_(-stdv, stdv)

  def forward(self, inputs):
    if type(inputs) is not tuple:
      inputs = tuple([inputs])
    if len(inputs) == 1:
      inputs = tuple([inputs[0], Variable(inputs[0].data.new(*inputs[0].size()).zero_())])
      
    inputs_cat = torch.cat(inputs, dim=1)
    g = F.sigmoid(self.lin(inputs_cat))
    z0, z1, r, q = torch.chunk(g, chunks=4, dim=1)
    h0_cap = torch.cat([inputs[0], inputs[1]*r], dim=1)
    h1_cap = torch.cat([inputs[1], inputs[0]*q], dim=1)                   
    h0_cap = F.tanh(self.gate0(h0_cap))
    h1_cap = F.tanh(self.gate1(h1_cap))

    return z0*h1_cap + (1.-z0)*inputs[0], z1*h0_cap + (1.-z1)*inputs[1]


class PSLRUCell(nn.Module):
  '''
  Decoupled Projected State

  LRU is 2-dimensional, hence it has 2 inputs and 2 outputs
  
  Args:
    hidden_size: The number of features in the hidden state h (also called the size of LRU),
                also number of features of the input (both dimensions have the same input)
    bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True

  Inputs: (h1,h2) or (h1) or h1
    - **h1** (batch, hidden_size): tensor containing h1 features
    - **h2** (batch, hidden_size): (optional; will be set to zero if not provided) 
                                   tensor containing h2 features

  Outputs: h1, h2
    - **h1** (batch, hidden_size): tensor containing the next h1
    - **h2** (batch, hidden_size): tensor containing the next h2
  '''

  def __init__(self, hidden_size, bias=True):
    super(PSLRUCell, self).__init__()
    self.lin = nn.Linear(hidden_size*2, hidden_size*2, bias=bias)
    self.gate0 = nn.Linear(hidden_size*2, hidden_size)
    self.gate1 = nn.Linear(hidden_size*2, hidden_size)

    self.reset_parameters([self.lin, self.gate0, self.gate1], hidden_size)
    
  def reset_parameters(self, module_list, hidden_size):
    ## glorot initialization
    stdv = 1. / math.sqrt(hidden_size)
    for module in module_list:
      for i, param in enumerate(module.parameters()):
        param.data.uniform_(-stdv, stdv)

    
  def forward(self, inputs):
    if type(inputs) is not tuple:
      inputs = tuple([inputs])
    if len(inputs) == 1:
      inputs = tuple([inputs[0], Variable(inputs[0].data.new(*inputs[0].size()).zero_())])
      
    inputs_cat = torch.cat(inputs, dim=1)
    g = F.sigmoid(self.lin(inputs_cat))
    z, r = torch.chunk(g, chunks=2, dim=1)
    h0_cap = torch.cat([inputs[0], inputs[1]*r], dim=1)
    h1_cap = torch.cat([inputs[1], inputs[0]*r], dim=1)                   
    h0_cap = F.tanh(self.gate0(h0_cap))
    h1_cap = F.tanh(self.gate1(h1_cap))

    return z*h1_cap + (1.-z)*inputs[0], z*h0_cap + (1.-z)*inputs[1]

## **Depreciated** Alternate to torch.chunk 
def chunk(mat, chunks):
  length = mat.size(1)
  chunk_length = length/chunks
  return ( mat[:,i:j] for i,j in zip(range(0,length,chunk_length), range(chunk_length, length+1, chunk_length)))

class LRUxCell(nn.Module):
  '''
  An LRUxCell is a composite of LRU cells stacked over each other. The concept is 
  similar to Recurrent Highway Unit
  '''
  def __init__(self, hidden_size, num_layers, bias=True, **kwargs):
    super(LRUxCell, self).__init__()

    ## choose from the set of lru units
    if 'unit' not in kwargs:
      kwargs['unit'] = LRUCell
      
    self.rnn = nn.ModuleList([kwargs['unit'](hidden_size, bias=bias) for _ in range(num_layers)])

  def forward(self, inputs):
    for cell in self.rnn:
      inputs = cell(inputs)
    return inputs

class HIGHWAYxCell(nn.Module):
  '''
  A Highway Cell
  '''
  def __init__(self, hidden_size, num_layers=1, bias=True):
    super(HIGHWAYxCell, self).__init__()
    self.num_layers = num_layers
    self.lin = nn.ModuleList([nn.Linear(hidden_size*2, hidden_size, bias=bias)]+
                             [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
    self.gate = nn.ModuleList([nn.Linear(hidden_size*2, hidden_size, bias=bias)]+
                              [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
  def forward(self, inputs):
    if type(inputs) is not tuple:
      inputs = tuple([inputs])
    if len(inputs) == 1:
      inputs = tuple([inputs[0], Variable(inputs[0].data.new(*inputs[0].size()).zero_())])
      
    inputs_cat = torch.cat(inputs, dim=1)
    h = F.tanh(self.lin[0](inputs_cat))
    t = F.sigmoid(self.gate[0](inputs_cat))
    s = h*t + inputs[1]*(1.-t)
    
    for layer in range(1, self.num_layers):
      h = F.tanh(self.lin[layer](s))
      t = F.sigmoid(self.gate[layer](s))
      s = h*t + s*(1.-t)

    return s,s

class Lattice(nn.Module):
  '''
  Args:
    hidden_size: The number of features in the hidden state h
    num_layers: Number of recurrent layers.
    bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    lru_layers: number of layers in the LRUCell (for highway-like unit)
    
    #TODO
    tied: If true, the weights will be tied
    dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
    bidirectional: If True, becomes a bidirectional RNN. Default: False


  Inputs: (h1,h2) or (h1) or h1
    - **h1** (seq_len, batch, hidden_size): tensor containing h1 features 
                                            (supports torch.nn.utils.rnn.PackedSequence as well)
    - **h2** (seq_len, batch, hidden_size): (optional; will be set to zero if not provided) 
                                   tensor containing h2 features

  Outputs: h1_n, h2_n
    - **h1_n** (seq_len, batch, hidden_size): tensor containing the next h1
    - **h2_n** (seq_len, batch, hidden_size): tensor containing the next h2
    :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output will also be a
      packed sequence.

  '''
  def __init__(self, hidden_size, num_layers, lru_layers=2, bias=True, batch_first=False, dropout=False, bidirectional=False, cell=LRUxCell, **kwargs):
    super(Lattice, self).__init__()
    self.layers = nn.ModuleList([cell(hidden_size, bias=bias, num_layers=lru_layers, **kwargs) for _ in range(num_layers)])
    self.batch_first = batch_first
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    
  def forward(self, H0, H1=None):
    ## H0 can be a packed sequence, while H1 can't be
    is_packed = isinstance(H0, PackedSequence)      
    if is_packed:
      H0, batch_sizes = H0
    else:
      dim = 0 if self.batch_first else 1
      batch_sizes = [H0.size(dim)]*H0.size(1-dim)
    
    if H1 is None:
      sizes = (batch_sizes[0], self.num_layers, self.hidden_size)
      H1 = Variable(H0.data.new(*sizes).zero_())

    T = len(batch_sizes)
    L = H1.size(1)
    H0_clone = H0.clone()
    start = 0
    for t in range(T):
      delta = batch_sizes[t]
      if is_packed:
        h0 = H0[start:start+delta,:]
      else:
        h0 = H0[:delta,t,:]
      H1_clone = H1.clone()
      for l in range(L):
        h1 = H1[0:delta,l,:]
        h0, h1 = self.layers[l]((h0,h1))
        H1_clone[0:delta,l,:] = h1
      H1 = H1_clone
      if is_packed:
        H0_clone[start:start+delta,:] = h0
      else:
        H0_clone[:delta,t,:] = h0
      start = start+delta
    H0 = H0_clone

    if not self.batch_first:
      H1 = H1.transpose(0,1)
    
    if is_packed:
      H0 = PackedSequence(H0, batch_sizes)

    return H0, H1

  
class GridLSTM(nn.Module):
  '''
  Args:
    hidden_size: The number of features in the hidden state h
    num_layers: Number of recurrent layers.
    bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    lru_layers: number of layers in the LRUCell (for highway-like unit)

    #TODO
    tied: If true, the weights will be tied
    dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
    bidirectional: If True, becomes a bidirectional RNN. Default: False


  Inputs: (h1,h2) or (h1) or h1
    - **h1** (seq_len, batch, hidden_size): tensor containing h1 features
                                            (supports torch.nn.utils.rnn.PackedSequence as well)
    - **h2** (seq_len, batch, hidden_size): (optional; will be set to zero if not provided) 
                                   tensor containing h2 features

  Outputs: h1_n, h2_n
    - **h1_n** (seq_len, batch, hidden_size): tensor containing the next h1
    - **h2_n** (seq_len, batch, hidden_size): tensor containing the next h2
    :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output will also be a
      packed sequence.

  '''
  def __init__(self, hidden_size, num_layers, lru_layers=2, bias=True, batch_first=False, dropout=False, bidirectional=False):
    super(GridLSTM, self).__init__()
    self.layers = nn.ModuleList([
      nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size, bias=bias),
                    nn.LSTMCell(hidden_size, hidden_size, bias=bias)])
      for _ in range(num_layers)])
    self.batch_first = batch_first
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    
  def forward(self, H0, mem=None):
    ## H0 can be a packed sequence, while H1 can't be
    is_packed = isinstance(H0, PackedSequence)      
    if is_packed:
      H0, batch_sizes = H0
    else:
      dim = 0 if self.batch_first else 1
      batch_sizes = [H0.size(dim)]*H0.size(1-dim)

    if isinstance(mem, tuple):
      H1 = mem[0]
      M0 = mem[1]
      M1 = mem[2]
    elif mem is None:
      H1 = None
      M0 = None
      M1 = None
    else:
      assert 0, 'Unsupported input for memory variables'
      
    if M0 is None:
      sizes = H0.size()
      M0 = Variable(H0.data.new(*sizes).zero_())
      
    if H1 is None:
      sizes = (batch_sizes[0], self.num_layers, self.hidden_size)
      H1 = Variable(H0.data.new(*sizes).zero_())

    if M1 is None:
      sizes = (batch_sizes[0], self.num_layers, self.hidden_size)
      M1 = Variable(H0.data.new(*sizes).zero_())

    T = len(batch_sizes)
    L = H1.size(1)
    H0_clone = H0.clone()
    M0_clone = M0.clone()
    start = 0
    for t in range(T):
      delta = batch_sizes[t]
      if is_packed:
        h0 = H0[start:start+delta,:]
        m0 = M0[start:start+delta,:]
      else:
        h0 = H0[:delta,t,:]
        m0 = M0[:delta,t,:]
      H1_clone = H1.clone()
      M1_clone = M1.clone()
      for l in range(L):
        h1 = H1[0:delta,l,:]
        m1 = M1[0:delta,l,:]
        h0, m0 = self.layers[l][0](h1, (h0,m0))
        h1, m1 = self.layers[l][1](h0, (h1,m1))
        H1_clone[0:delta,l,:] = h1
        M1_clone[0:delta,l,:] = m1
      H1 = H1_clone
      M1 = M1_clone
      if is_packed:
        H0_clone[start:start+delta,:] = h0
        M0_clone[start:start+delta,:] = m0
      else:
        H0_clone[:delta,t,:] = h0
        M0_clone[:delta,t,:] = m0
      start = start+delta
    H0 = H0_clone
    M0 = M0_clone

    if not self.batch_first:
      H1 = H1.transpose(0,1)
      M1 = M1.transpose(0,1)
      
    if is_packed:
      H0 = PackedSequence(H0, batch_sizes)
      M0 = PackedSequence(M0, batch_sizes)
    return H0,(H1,M0,M1)
      
## Model chooser
def rnn_chooser(model, **kwargs):
  if model=='lru':
    return Lattice(kwargs['hidden_size'],
                   kwargs['num_layers'],
                   lru_layers=kwargs['lru_layers'],
                   batch_first=True,
                   cell=LRUxCell,
                   unit=LRUCell)
  elif model=='rglru':
    return Lattice(kwargs['hidden_size'],
                   kwargs['num_layers'],
                   lru_layers=kwargs['lru_layers'],
                   batch_first=True,
                   cell=LRUxCell,
                   unit=RGLRUCell)
  elif model=='pslru':
    return Lattice(kwargs['hidden_size'],
                   kwargs['num_layers'],
                   lru_layers=kwargs['lru_layers'],
                   batch_first=True,
                   cell=LRUxCell,
                   unit=PSLRUCell)
  elif model=='glstm':
    return GridLSTM(kwargs['hidden_size'],
                   kwargs['num_layers'],
                   batch_first=True)
  elif model=='highway':
    return Lattice(kwargs['hidden_size'],
                   kwargs['num_layers'],
                   lru_layers=kwargs['lru_layers'],
                   batch_first=True,
                   cell=HIGHWAYxCell)
  elif model=='gru':
    return nn.GRU(kwargs['hidden_size'],
                  kwargs['hidden_size'],
                  kwargs['num_layers'],
                  batch_first=True)
  elif model=='lstm':
    return nn.LSTM(kwargs['hidden_size'],
                   kwargs['hidden_size'],
                   kwargs['num_layers'],
                   batch_first=True)
  else:
    assert 0, 'Model `%s` not found'%(model)

## Models
class langModel(nn.Module):
  '''
  Generic language model. 
  Requires a RNNclass and corresponding **kwargs as inputs
  '''
  def __init__(self, model='lru', **kwargs):
    super(langModel, self).__init__()
    self.affine_enc = nn.Embedding(kwargs['vocab_size'], kwargs['hidden_size'])
    self.rnn = rnn_chooser(model, **kwargs)
    self.affine_dec = nn.Linear(kwargs['hidden_size'], kwargs['vocab_size'])
  
  def forward(self, inputs, mem=None):
    sizes = inputs.size()
    inputs = self.affine_enc(inputs)

    outputs, mem = self.rnn(inputs, mem)

    ## remove memory variables from graph,
    ## as they will be used as input for the next batch
    if type(mem) is tuple:
      mem = tuple([m.detach() for m in mem])
    else:
      mem.detach_()
    outputs = self.affine_dec(outputs.contiguous().view(-1, outputs.size(-1)))
    return outputs, mem
