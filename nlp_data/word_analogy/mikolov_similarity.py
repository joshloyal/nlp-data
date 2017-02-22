from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import codecs
import collections
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

from nlp_data.base import fetch_dataset, get_data_home, load_file, DataBundle


URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt"
ARCHIVE_NAME = "word-test.v1.txt"
CACHE_NAME = "word-test.v1.pkz"


def load_file(file_name):
    """Load the word similarity files in Mikolov et. all 2013.
    The file begins with a header (starting with //) and then
    each topic is deliniated by a line of the form : topic name.
    The actually analogy tasks are four words seperated by
    spaces.
    """
    topics = collections.defaultdict(list)
    current_topic = None
    with open('word-test.v1.txt', 'rb') as f:
        for line in f:
            if line.startswith('//'):
                continue
            elif line.startswith(':'):
                current_topic = line.strip()[2:]
                continue
            topics[current_topic].append(line.strip().split())

    # convert lists to numpy arrays
    for key, analogies in topics.iteritems():
        topics[key] = np.array(analogies)

    return dict(topics)


def download_mikolov_similarity(target_dir, cache_path):
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    opener = urllib2.urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(opener.read())

    cache = load_file(archive_path)

    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)

    return cache



def make_dataset(dataset, categories=None):
    if categories is None:
        categories = dataset.keys()

    X = []
    for category in categories:
        if category not in dataset:
            raise ValueError('Unrecognized category: {}'.format(category))
        X.append(dataset[category])

    return np.vstack(X)



def fetch_mikolov_similarity(categories=None):
    """fetch_mikolov_similarity

    Parameters
    ----------
    categories : None or collection of string or unicode
        If None (default), load all the categories.
        If not None, list of category names to load
        (other categories ignored). The possible categories are
            - capital-common-countries
            - capital-world
            - city-in-state
            - currency
            - family
            - gram1-adjective-to-adverbe
            - gram2-opposite
            - gram3-comparative
            - gram4-superlative
            - gram5-present-participle
            - gram6-nationality-adjective
            - gram7-past-tense
            - gram8-plural
            - gram9-plural-verbs
    """
    data_home = get_data_home()
    cache_path = os.path.join(data_home, CACHE_NAME)

    mikolov_home = os.path.join(data_home, 'mikolov_home')
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
        cache = download_mikolov_similarity(target_dir=mikolov_home,
                                            cache_path=cache_path)

    return make_dataset(cache, categories)
