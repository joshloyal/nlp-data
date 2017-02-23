from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import codecs
import glob
import os
import tarfile
import urllib2
import shutil
import six
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets.base import load_files

from nlp_data import base


URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_NAME = "aclImdb_v1.tar.gz"
CACHE_NAME = "aclImdb_v1.pkz"
DATASET_FOLDER = "aclImdb"


def download_imdb_reviews(target_dir, cache_path):
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    opener = urllib2.urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(opener.read())

    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)

    train_path = os.path.join(target_dir, DATASET_FOLDER, 'train')
    test_path = os.path.join(target_dir, DATASET_FOLDER, 'test')
    train_data = load_files(train_path, categories=['pos', 'neg'])
    test_data = load_files(test_path, categories=['pos', 'neg'])

    cache = dict(train=base.to_databundle(train_data),
                 test=base.to_databundle(test_data))

    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)

    return cache


def fetch_imdb_reviews(as_onehot=False, train_test_split=False):
    """fetch_imdb_reviews

    Parameters
    ----------
    as_onehot : bool
        Whether the target should be binary encoded

    train_test_split : bool
        Whether to perform a train/test split
    """
    data_home = base.get_data_home()
    cache_path = os.path.join(data_home, CACHE_NAME)

    movie_home = os.path.join(data_home, 'imdb_reviews_home')
    cache = None

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                    compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if cache is None:
        cache = download_imdb_reviews(target_dir=movie_home,
                                      cache_path=cache_path)

    return base.fetch_dataset(cache,
                              train_test_split=train_test_split,
                              as_onehot=as_onehot)
