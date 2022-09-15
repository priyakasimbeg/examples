"""Contains Tokenizer class for word level tokenization.
"""

import tensorflow as tf
import os

# from ml_collections.config_dict import config_dict

# BEGIN GOOGLE-INTERNAL
# TODO(b/244225355): Move data to placer
DATA_DIR = os.path.expanduser('~/pytorch/fork/examples/word_language_model/data/wikitext-2')
# END GOOGLE-INTERNAL
TRAIN_FILENAME = 'train.txt'
VALID_FILENAME = 'valid.txt'
TEST_FILENAME = 'test.txt'
BUFFER_SIZE = 5_000_000
PAD_ID = -1

EOS_TOKEN = b'<eos>'
UNKNOWN_TOKEN = b'<unk>'


class _Dictionary:
  """Dictionary contains word-to-id mappings and id-to-word mappings.

  Attributes:
    word2idx: dict containing key-values where keys are words and values are
      tokens.
    idx2word: list where the index of each word in the list is the token value.
  """

  def __init__(self):
    self.word2idx = {}
    self.idx2word = []

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
    return self.word2idx[word]

  def __len__(self):
    return len(self.idx2word)


class Tokenizer:
  """Tokenizer object for word level tokenization from words to unique ids.

  Attributes:
    dictionary: Dictionary containing word-to-id and id-to-word mappings
  """

  def __init__(self):
    self.dictionary = _Dictionary()

  def train(self, dataset: tf.data.TextLineDataset):
    """Trains a Tokenizer from a TextLineDataset."""
    # Add words to the dictionary
    for line in dataset:
      words = line.numpy().split() + [EOS_TOKEN]
      for word in words:
        self.dictionary.add_word(word)

  def tokenize(self, dataset: tf.data.TextLineDataset) -> tf.data.Dataset:
    """Tokenizes a TextLineDataset."""
    idss = []
    for line in dataset:
      ids = []
      words = line.numpy().split() + [b'<eos>']
      for word in words:
        try:
          ids.append(self.dictionary.word2idx[word])
        except KeyError:
          ids.append(self.dictionary.word2idx[UNKNOWN_TOKEN])
      idss.append(ids)
    ids = tf.concat(idss, 0)

    tokenized_dataset = tf.data.Dataset.from_tensor_slices(ids)

    return tokenized_dataset


    """Module for processing wikitext-2 train, val and test datasets from raw text files to tokenized and batched tensorflow.data.Datasets."""


# TODO(b/240435836): Add manual workflow for tokenizers for LM datasets
def get_trained_tokenizer(train_dataset: tf.data.Dataset,) -> tf.data.Dataset:
  tokenizer = Tokenizer()
  tokenizer.train(train_dataset)
  return tokenizer


def split_input_target(sequence):
  input_sequence = sequence[:-1]
  target_sequence = sequence[1:]
  return {'inputs': input_sequence, 'targets': target_sequence}


def get_wikitext2_dataset(train_batch_size: int,
                          valid_batch_size: int, 
                          test_batch_size: int,
                          sequence_length: int, 
                          shuffle_seed: int) -> tf.data.Dataset:
  """Returns wikitext-2 dataset.

  Args:
    hps: Dataset hyper parameters.
    train_batch_size: Batch size for train iterations
    valid_batch_size: Batch size for validation iterations
    test_batch_size: Batch size for test iterations
    shuffle_seed: seed for shuffling dataset sequences

  Returns:
    train_dataset, valid_dataset, test_dataset
  """
  train_path = os.path.join(DATA_DIR, TRAIN_FILENAME)
  valid_path = os.path.join(DATA_DIR, VALID_FILENAME)
  test_path = os.path.join(DATA_DIR, TEST_FILENAME)

  # Get TextLineDataset from raw files
  train_text_dataset = tf.data.TextLineDataset(train_path)
  valid_text_dataset = tf.data.TextLineDataset(valid_path)
  test_text_dataset = tf.data.TextLineDataset(test_path)

  # TODO(b/240435836): Add manual workflow for tokenizers for LM datasets
  # Tokenize data
  tokenizer = get_trained_tokenizer(train_text_dataset)

  train_dataset_tokenized = tokenizer.tokenize(train_text_dataset)
  valid_dataset_tokenized = tokenizer.tokenize(valid_text_dataset)
  test_dataset_tokenized = tokenizer.tokenize(test_text_dataset)

  # Divide data in sequences.
  train_dataset_sequences = train_dataset_tokenized.batch(
      sequence_length + 1,
      drop_remainder=True)
  valid_dataset_sequences = valid_dataset_tokenized.padded_batch(
      sequence_length + 1,
      drop_remainder=True)
  test_dataset_sequences = test_dataset_tokenized.padded_batch(
      sequence_length + 1,
      drop_remainder=True)

  # Split the sequences into inputs and targets.
  train_dataset_sequences = train_dataset_sequences.map(split_input_target)
  valid_dataset_sequences = valid_dataset_sequences.map(split_input_target)
  test_dataset_sequences = test_dataset_sequences.map(split_input_target)

  # Shuffle the train sequences.
  train_dataset_sequences = train_dataset_sequences.shuffle(
      BUFFER_SIZE, seed=shuffle_seed)

  # Copy the train_dataset_sequences to a non repeating dataset
  eval_train_dataset_sequences = train_dataset_sequences

  # Perform batching for training, validation and testing.
  # Make training data repeat indefinitely.
  train_dataset_sequences = train_dataset_sequences.repeat()
  train_dataset = train_dataset_sequences.batch(
      train_batch_size,
      drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  # Use padded batches for eval_train, validation and test_datasets since the
  # sequences do not repeat indefintely.
  eval_train_dataset = eval_train_dataset_sequences.batch(
      train_batch_size,
      drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  valid_dataset = valid_dataset_sequences.padded_batch(
      valid_batch_size,
      drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset_sequences.padded_batch(
      test_batch_size,
      drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

  return train_dataset, eval_train_dataset, valid_dataset, test_dataset