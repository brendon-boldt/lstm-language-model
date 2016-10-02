import sys
import argparse
import collections
import tensorflow as tf
import numpy as np

import reader

parser = argparse.ArgumentParser(
    description="Prepare data for the TensorFlow LSTM RNN")
parser.add_argument('infile')
parser.add_argument('-o', '--out_dir', nargs='?', default='./')
parser.add_argument('-v', '--vocab_size', type=int, nargs='?', default=10000)
parser.add_argument('-f', '--frequency', action='store_true')
args = parser.parse_args()
args.out_dir = args.out_dir.rstrip('/')+'/'

'''
- In the PTB <unk> was the second most common token
'''

def _get_word_frequency():
  data = reader._read_words(args.infile)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  string = ""
  for v in count_pairs[:100]:
    #string += (str(v[0]) + "\n" +  str(v[1]/len(data))) + "\n\n"
    string += (str(v[1]/len(data)) + "\t" + str(v[0])) + "\n"
  return string

def _get_freq_frequency(data):
  data = reader._read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  cc = collections.Counter([x[1] for x in count_pairs])
  for k in cc:
    print(str(k) + "\t" + str(cc[k]/len(count_pairs)))

def _next_sentence_index(data, start):
  for index in range(start, len(data)):
    #if (data[index] == "<eos>"):
    if (data[index] == "\n"):
      return index+1
  return None

def _divide_data(train_prop = 0.8):
  with tf.gfile.GFile(args.infile, "r") as f:
    #raw_data = f.read().replace("\n", "<eos>").split(" ")
    raw_data = f.read().split(" ")
  size = len(raw_data)

  '''
  counter = collections.Counter(raw_data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  vocab = set([count_pairs[i][0] for i in range(0, min(len(count_pairs),args.vocab_size))])

  for i in range(0, size):
    if not raw_data[i] in vocab:
      raw_data[i] = '<unk>'
  '''

  # We want to take the test and validation data from the middle of
  # the dataset just in case the beginning or end has anomolies
  train_index = _next_sentence_index(raw_data, round(size*train_prop/2))
  valid_index = _next_sentence_index(raw_data, round(size/2))
  end_index = _next_sentence_index(raw_data, round(size-(size*train_prop/2)))

  train_data = raw_data[:train_index] + raw_data[end_index:]
  valid_data = raw_data[train_index:valid_index]
  test_data  = raw_data[valid_index:end_index]


  counter = collections.Counter(train_data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  # len(count_pairs) - 1 because if we do not have at least one <unk> token
  # in the training set, that well cause problems for the test/valid set
  actual_size = min(args.vocab_size, len(count_pairs)-1)
  vocab = set([count_pairs[i][0] for i in range(0, actual_size)])

  for i in range(0, len(train_data)):
    if not train_data[i] in vocab:
      train_data[i] = '<unk>'
  for i in range(0, len(valid_data)):
    if not valid_data[i] in vocab:
      valid_data[i] = '<unk>'
  for i in range(0, len(test_data)):
    if not test_data[i] in vocab:
      test_data[i] = '<unk>'

  return train_data, valid_data, test_data

def _write_data(train_data, valid_data, test_data):
  with tf.gfile.GFile(args.out_dir + "train.txt", "w") as f:
    f.write(' '.join(train_data))
  with tf.gfile.GFile(args.out_dir + "valid.txt", "w") as f:
    f.write(' '.join(valid_data))
  with tf.gfile.GFile(args.out_dir + "test.txt", "w") as f:
    f.write(' '.join(test_data))

if (args.frequency):
  print(_get_word_frequency())
else:
  raw_data = _divide_data()
  _write_data(*raw_data)
