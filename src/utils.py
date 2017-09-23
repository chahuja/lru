import numpy as np
import os
import warnings

class Name(object):
  ''' Create a name based on hyper-parameters, other arguments 
  like number of epochs or error rates
  
  Format:
  path2file/...argname_value_..._outputkind.ext

  args: Namespace(argname,value, ....) generally taken from the argparser
    argname: Hyper-parameters (i.e. model structure)
    value: Values of the corresponding Hyper-parameters

  path2file: set as './' by default and decides the path where the file is to be stored
  outputkind: what is the kind of output 'err', 'vis', 'cpk' or any other acronym given as a string
  ext: file type given as a string

  *args_subset: The subset of arguments to be used and its order
  '''
  def __init__(self, args, *args_subset):
    self.name = ''
    args_dict = vars(args)

    ## If args_subset is empty exit
    assert args_subset, 'Subset of arguments to be chosen is empty'
    
    ## Scan through required arguments in the name
    for arg in args_subset:
      if (arg not in args_dict):
        warnings.warn('Key %s does not exist. Skipping...'%(arg))
      else:
        self.name += '%s_%s_' % (''.join(arg.split('_')), '-'.join(str(args_dict[arg]).split('.')))


  def __call__(self, outputkind, ext, path2file='./'):
    try:
      os.makedirs(path2file)
    except OSError:
      if not os.path.isdir(path2file):
        raise 'Directory could not be created. Check if you have the required permissions to make changes at the given path.'
    return os.path.join(path2file,self.name + '%s.%s' %(outputkind,ext))  
