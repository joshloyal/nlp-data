import os

import numpy as np
import pandas as pd

from nlp_data import array_utils


class DataBundle(object):
    def __init__(self, data, target, filenames=[], description=''):
        self.data = data
        self.target = target
        self.filenames = filenames
        self.description = description

    def __repr__(self):
        data_shape = array_utils.get_shape(self.data)
        return ('%s(n_samples=%d, n_features=%d, n_classes=%d)' % (
                    self.__class__.__name__,
                    data_shape[0],
                    data_shape[1],
                    np.unique(self.target).shape[0])
                )


def get_data_home(data_home=None):
    if data_home is None:
        data_home = os.environ.get('NLP_DATA',
                                   os.path.join('~', 'nlp_data'))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home


def download_and_untar(target_path, url):
    opener = urllib2.urlopen(url)
    with open(target_path, 'wb') as f:
        f.write(opener.read())

    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)


def load_file(file_name, encoding='utf-8'):
    with open(file_name, 'r') as f:
        return [line.decode(encoding).strip() for line in f]


def fetch_dataset(dataset, train_test_split=False, as_onehot=False):
    dataset_train = dataset['train']
    dataset_test = dataset['test']

    x_train = np.asarray(dataset_train.data)
    x_test = np.asarray(dataset_test.data)
    y_train = np.asarray(dataset_train.target)
    y_test = np.asarray(dataset_test.target)
    n_classes = np.max(y_train) + 1

    if as_onehot:
        y_train = array_utils.to_categorical(y_train, n_classes=n_classes)
        y_test = array_utils.to_categorical(y_test, n_classes=n_classes)

    if not train_test_split:
        return (np.concatenate((x_train, x_test), axis=0),
                np.concatenate((y_train, y_test), axis=0),
                n_classes)

    return (x_train, y_train), (x_test, y_test), n_classes
