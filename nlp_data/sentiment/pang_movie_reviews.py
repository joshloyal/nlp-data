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

from nlp_data.base import fetch_dataset, get_data_home, load_file, DataBundle


URL = ("http://www.cs.cornell.edu/people/pabo/movie-review-data/"
       "rt-polaritydata.tar.gz")
ARCHIVE_NAME = "rt-polaritydata.tar.gz"
CACHE_NAME = "rt-polaritydata.pkz"
DATASET_FOLDER = "rt-polaritydata"


def download_movie_reviews(target_dir, cache_path):
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    opener = urllib2.urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(opener.read())

    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)

    pos_path = glob.glob(os.path.join(target_dir, DATASET_FOLDER, '*.pos'))[0]
    neg_path = glob.glob(os.path.join(target_dir, DATASET_FOLDER, '*.neg'))[0]
    pos_data = load_file(pos_path, label=1, encoding='latin1')
    neg_data = load_file(neg_path, label=0, encoding='latin1')
    dataset = pd.concat([pos_data, neg_data])
    X, y = dataset['data'].values, dataset['target'].values
    cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=1234)
    train, test = next(cv.split(y))

    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]

    cache = dict(train=DataBundle(data=X_train, target=y_train),
                 test=DataBundle(data=X_test, target=y_test))

    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)

    return cache


def fetch_movie_reviews(as_onehot=False, train_test_split=False):
    """pang_movie_reviews

    Parameters
    ----------
    as_onehot : bool
        Whether the target should be binary encoded

    train_test_split : bool
        Whether to perform a train/test split
    """
    data_home = get_data_home()
    cache_path = os.path.join(data_home, CACHE_NAME)
    movie_home = os.path.join(data_home, 'movie_reviews_home')
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
        cache = download_movie_reviews(target_dir=movie_home,
                                       cache_path=cache_path)

    return fetch_dataset(cache,
                         train_test_split=train_test_split,
                         as_onehot=as_onehot)
