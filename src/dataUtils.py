from __future__ import print_function
import numpy as np
import os
from collections import Counter
import json
from codecs import open
import warnings
import subprocess
import pdb
import sys
from tqdm import tqdm

class Data(object):
  '''
  Creates an iterator for a language modeling task
  The first object of this model has to be a train

  file_path: path to the data file
  suffix: `.train`, `.val` or `.test`
  '''
  
  def __init__(self,
               file_path,
               suffix,
               encoding='utf-8',
               batch_size = 100,
               num_unrolling=50,
               w2v=False,
               delimiter='',
               max_vocab=0,
               valid_frac=0.1,
               test_frac=0.1,
               vocab=None,
               rev_vocab=[]):
    
    self._UNK = '_UNK'
    
    self.file_path = file_path + suffix
    self.encoding = encoding
    self.batch_size = batch_size
    self.num_unrolling = num_unrolling
    self.delimiter = delimiter
    self.w2v = w2v
    self.vocab = {self._UNK:0} 
    self.rev_vocab = [self._UNK]
    self.max_vocab = max_vocab

    
    ## split data
    if not os.path.exists(self.file_path):
      self.split_data(file_path, valid_frac, test_frac)
    elif suffix == '.train':
      print('Train, val and test splits already exist')
  
    ## pre-batching checks like vocabulary, number of lines
    if not w2v and vocab is None:
      num_samples = self.create_vocab(self.file_path, self.vocab, self.rev_vocab)
      print('Vocabulary Size: %d'%(len(self.rev_vocab)))
      print('Number of tokens: %d'%(num_samples))

    elif vocab is not None:
      self.vocab=vocab
      self.rev_vocab=rev_vocab
    else:
      '''
      use the w2v function here
      '''
      assert 0, 'word2vec function not implemented'

  def tokenizer(self, string, delimiter):
    if delimiter == '':
      return list(string)
    else:
      return string.split(delimiter)
    
  def split_data(self, file_path, valid_frac, test_frac):
    with open(file_path,'r', encoding=self.encoding) as f:
      raw = f.read()
    f.close()
    raw_len = len(raw)
    val_size = int(valid_frac*raw_len)
    test_size = int(test_frac*raw_len)
    train_size = raw_len - val_size - test_size

    train = raw[:train_size]
    val = raw[train_size:train_size+val_size]
    test = raw[train_size+val_size:]
    def file_write(file_path, data, suffix):
      with open(file_path+suffix, 'w', encoding=self.encoding) as f:
        f.write(data)
    file_write(file_path, train, '.train')
    file_write(file_path, val, '.val')
    file_write(file_path, test, '.test')
    
  def create_vocab(self, file_path, vocab, rev_vocab):
    num_samples = 0
    wordcount = Counter()
    if not os.path.isfile(file_path+'.vocab'):
      print('Vocabulary does not exist. Creating...')
      with open(file_path, encoding=self.encoding) as f:
        raw = f.read()      
      line = self.tokenizer(raw, self.delimiter)

      wordcount.update(line)
      num_samples=len(line)
      if self.max_vocab > 0:
        vocab_temp = wordcount.most_common()[:self.max_vocab]
      else:
        vocab_temp = wordcount.most_common()
      for i, (token, count) in enumerate(vocab_temp):
        vocab.update({token:i+1}) # hardcoded to include _UNK by default
        rev_vocab.append(token)
      
      with open(file_path+'.vocab', 'w', encoding=self.encoding) as f:
        json.dump({'0':vocab, '1':rev_vocab, '2': num_samples}, f)
    else:
      print('Vocabulary exists. Loading...')
      vocab.clear()
      rev_vocab[:] = []
      with open(file_path+'.vocab', encoding=self.encoding) as f:
        raw_vocab = json.load(f)
        vocab.update(raw_vocab['0'])
        rev_vocab += raw_vocab['1']
        num_samples = raw_vocab['2']
    return num_samples
  
  def add2vocab(self, vocab, rev_vocab, line):
    for token in line:
      if token not in vocab:
        vocab[token] = len(rev_vocab)
        rev_vocab.append(token)

  def token2id(self, token):
    if token in self.vocab:
      return self.vocab[token]
    else:
      return self.vocab[self._UNK]
  
  def id2token(self, iD):
    if iD >= len(self.rev_vocab) or iD < 0:
      return self._UNK
    else:
      return self.rev_vocab[int(iD)]
    pass

  def sample(self, distribution):
    r = np.random.uniform(0, 1, size=(distribution.shape[0]))
    s = np.zeros((distribution.shape[0]))
    indices = np.zeros((distribution.shape[0]), dtype=np.int)
    
    for i in range(distribution.shape[1]):
      s += distribution[:,i]
      indices += np.asarray(s<=r, dtype=np.int)
    return indices

  def token2line(self, line):
    return self.delimiter.join([self.id2token(id) for id in line])

  def __iter__(self):
    fp = open(self.file_path, encoding=self.encoding)
    raw = fp.read()
    fp.close()

    raw = self.tokenizer(raw, self.delimiter)
    num_seq_per_batch = ((len(raw) // self.batch_size) - 1) // self.num_unrolling
    raw_len = ((num_seq_per_batch * self.num_unrolling) + 1) * self.batch_size
    raw = raw[:raw_len]
    
    len_per_batch = raw_len//self.batch_size
    chunk_size = self.num_unrolling + 1
      
    flag = True
    pointer = 0

    while(flag):
      line = []
      for i in range(self.batch_size):
        line += raw[i*len_per_batch+pointer:i*len_per_batch + pointer + chunk_size]
      if i*len_per_batch + pointer + chunk_size == len(raw):
        flag = False

      pointer = pointer + chunk_size - 1 ## induce an overlap of one token

      Range = range(0, len(line), self.num_unrolling+1)
      lines = [line[start:start+self.num_unrolling+1] for start in Range]
      batch = np.zeros((self.batch_size, self.num_unrolling+1), dtype=np.int)
      for index, value in enumerate(lines):
        for jndex, token in enumerate(value):
          try:
            batch[index, jndex] = self.token2id(token)
          except:
            pdb.set_trace()

      yield batch


'''
Depreciated
'''

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrolling, vocabulary_size, char2id):
    self._text = text
    self._batch_size = batch_size
    self._num_unrolling = num_unrolling
    self._vocabulary_size = vocabulary_size
    self._char2id = char2id

  def __call__(self):
    ## cut extra data at the end
    text_len = (len(self._text)//self._batch_size) * self._batch_size
    text_len = (((text_len/self._batch_size -1) // self._num_unrolling) * self._num_unrolling + 1) * self._batch_size
    self._text = self._text[:text_len]

    ## create batches
    batches = list()
    for step in range(self._num_unrolling + 1):
      segment = text_len/self._batch_size
      arr_len = text_len/self._num_unrolling
      arr = np.zeros((arr_len, self._vocabulary_size))
      sample_num = 0
      for c in range(step,segment-1, self._num_unrolling):
        for b in range(self._batch_size):
          arr[sample_num, self._char2id(self._text[b*segment+c])] = 1
          sample_num+=1
      batches.append(arr)

    return batches


class oldData(object):
  def __init__(self, filename, valid_frac=0.1, test_frac=0.1, batch_size=100, num_unrolling=10, criterion='char', delimiter=None):
    '''
    Data handler for character/word based models.
    criterion: char/word based spliting
    Note: for word based tokenisation, the following script is used https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
    '''

    assert valid_frac+test_frac <1, 'test_frac+valid_frac should be less than 1'

    self.unk_count = 0

    if(delimiter==None):
      if(criterion=='word'):
        ## search for existing tokenised file
        if(not os.path.isfile(filename + '.tokens')):
          subprocess.call('./tokenizer.perl -l en -no-escape' + ' < ' + filename + ' > ' + filename+'.tokens', shell=True)

        filename += '.tokens'
        delimiter = ' '
      elif(criterion=='char'):
        delimiter = ''
      else:
        assert (0), 'Incorrect criterion. Choose from char or word'
    
    ## read text from a given file
    text = self.read_data(filename)
    if(criterion=='word'):
      ## tokenzer.perl does not put spaces between escape sequences
      text = text.replace('\n',' \n ').replace('\n  \n','\n \n') ## hardcoded for shakespere dataset

    if(delimiter == ''):
      text = list(text)
    else:
      text=text.split(delimiter)

    self.delimiter = delimiter
      
    ## split text into train and test
    train_text, valid_text, test_text = self.split_data(text, valid_frac, test_frac)

    ## create vocabulary based on the data
    self.vocab, self.rev_vocab, self.vocabulary_size = self.create_vocab(train_text)

    train = BatchGenerator(train_text, batch_size, num_unrolling, self.vocabulary_size, self.char2id)
    val = BatchGenerator(valid_text, batch_size, 1, self.vocabulary_size, self.char2id)
    test = BatchGenerator(test_text, batch_size, 1, self.vocabulary_size, self.char2id)
    self.train_batches = train()
    self.val_batches = val()
    print('Unknown Tokens in val-set: %d') %(self.unk_count)
    self.unk_count = 0
    self.test_batches = test()
    print('Unknown Tokens in test-set: %d') %(self.unk_count)
    
  ## Read file as a single string
  def read_data(self, filename):
    f = open(filename, 'rb')
    data=f.read()
    f.close()
    return data

  ## Split given string into train and test
  def split_data(self, text, valid_frac, test_frac):
    """Split the given data into train and validation based on
    the valid_frac"""
    valid_size = int(np.ceil(len(text)*valid_frac))
    test_size = int(np.ceil(len(text)*test_frac)) ## hardcoded to 3 times valid frac
    valid_text = text[-valid_size:]
    test_text = text[-valid_size-test_size:-valid_size]
    train_text = text[:-valid_size-test_size]
    train_size = len(train_text)
    print(train_size, train_text[:64])
    print(valid_size, valid_text[:64])
    print(test_size, test_text[:64])
    return train_text, valid_text, test_text

  ## Given a string find a character/word based vocabulary
  def create_vocab(self, text):
    """Create a vocabulary for a given text and return 
    vocab, rev_vocab and vocabulary size
    """
    
    vocabulary_size = 1
    vocab = {'_UNK':0}
    rev_vocab = ['_UNK']
    for char in text:
      if char not in vocab.keys():
        vocab[char] = vocabulary_size
        rev_vocab += [char]
        vocabulary_size += 1
    print('Vocabulary Size: %d') % (vocabulary_size)
    return vocab, rev_vocab, vocabulary_size

  def char2id(self,char):
    """Convert a character to its correspoding id in the dictionary"""
    if char in self.vocab.keys():
      return self.vocab[char]
    else:
      self.unk_count += 1
      return 0

  def id2char(self,dictid):
    """Convert id(int) to the corresponding character"""
    if dictid >=0 :
      return self.rev_vocab[dictid]
    else:
      print ('Invalid ID: %d' % dictid)
      return 0

  def characters(self,probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [self.id2char(c) for c in np.argmax(probabilities, 1)]

  # Depreciated 
  def batches2string(self,batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
      s = [''.join(x) for x in zip(s, self.characters(b))]
    return s


## Sampling Functions

class Sample(object):
  def __init__(self, vocabulary_size):
    self.vocabulary_size = vocabulary_size
    
  def logprob(self, predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

  @staticmethod
  def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = np.random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
      s += distribution[i]
      if s >= r:
        return i
    return len(distribution) - 1

  def sample(self, prediction, batch_size=1):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[batch_size, self.vocabulary_size], dtype=np.float)
    for i in range(len(prediction)):
      index = self.sample_distribution(prediction[i,:])
      p[i, index] = 1.0
    return p

  
  def random_distribution(self, batch_size=1):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[batch_size, self.vocabulary_size])
    return b/np.sum(b, 1)[:,None]

