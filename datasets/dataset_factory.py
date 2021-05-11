"""
    Provide dataset given split name.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import reid
from datasets.utils import unpickle
import os
import tensorflow as tf

# provider functions might vary on different datasets
datasets_map = {
    'market1501': reid,
    'cifar100': reid
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
      name: String, the name of the dataset.
      split_name: A train/test split name.
      dataset_dir: The directory where the dataset files are stored.
      file_pattern: The file pattern to use for matching the dataset source files.
      reader: The subclass of tf.ReaderBase. If left as `None`, then the default
        reader defined by each dataset is used.

    Returns:
      A `Dataset` class.

    Raises:
      ValueError: If the dataset `name` is unknown.
    """
    print('==={}'.format(name))
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    # if name == 'cifar100':
    #   a = unpickle(dataset_dir + '/' + split_name)
    #   # print(a.keys())
    #   # print(a)
    #   data = a[b'data']     # numpy.ndarray
    #   print(data.shape)
    #   labels = a[b'coarse_labels']
    #   f_labels = a[b'fine_labels']      # list
    #   # print(len(labels))
    #   # print(max(labels))
    #   # print(len(f_labels))
    #   # print(max(f_labels))
    #   print(type(data))
    #   print(type(f_labels))
    #   ds = tf.data.Dataset.from_tensors([data, f_labels])
    #   print(list(ds.as_numpy_iterator())[0])
    #   input()
    #   return dataset
    return datasets_map[name].get_split(
        split_name,
        dataset_dir,
        file_pattern,
        reader)
