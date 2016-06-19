import sys
import collections
import tensorflow as tf
import numpy as np

import reader as re

def _get_word_frequency(data):
  data = re._read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  for v in count_pairs[:100]:
    print(str(v[0]) + "\n" +  str(v[1]/len(data)))
    print()

def _get_freq_frequency(data):
  data = re._read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  cc = collections.Counter([x[1] for x in count_pairs])
  for k in cc:
    print(str(k) + "\t" + str(cc[k]/len(count_pairs)))

def _next_sentence_index(data, start):
  for index in range(start, len(data)):
    if (data[index] == "<eos>"):
      return index+1
  return None

def _divide_data(filename, train_prop = 0.8):
  with tf.gfile.GFile(filename, "r") as f:
    raw_data = f.read().replace("\n", "<eos>").split(" ")
  size = len(raw_data)
  # We want to take the test and validation data from the middle of
  # the dataset just in case the beginning or end has anomolies
  train_index = _next_sentence_index(raw_data, round(size*train_prop/2))
  valid_index = _next_sentence_index(raw_data, round(size/2))
  end_index = _next_sentence_index(raw_data, round(size-(size*train_prop/2)))

  train_data = raw_data[:train_index] + raw_data[end_index:]
  valid_data = raw_data[train_index:valid_index]
  test_data  = raw_data[valid_index:end_index]

  print(len(train_data))
  print(len(valid_data))
  print(len(test_data))

filename = sys.argv[1]
_divide_data(filename)
