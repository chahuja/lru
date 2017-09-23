import torch
import pickle as pkl
import pdb

def save_model(model, file_path):
  f = open(file_path, 'wb') 
  pkl.dump(model.state_dict(), f)
  f.close()
    
def load_model(model, file_path):
  model.load_state_dict(pkl.load(open(file_path, 'rb')))

def accumulate_grads(model, grads_list):
  if grads_list:
    grads_list = [param.grad.data+old_grad.clone() for param, old_grad in zip(model.parameters(), grads_list)]
  else:
    grads_list += [param.grad.data for param in model.parameters()]
  return grads_list

def save_grads(val, file_path):
  pkl.dump(val, open(file_path, 'wb'))

def load_grads(file_path):
  return pkl.load(open(file_path))
