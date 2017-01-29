import os

import numpy as np
import pandas as pd


def to_categorical(target, n_classes=None):
    """Converts a target vector with integer labels to a binary class matrix.

    Parameters
    ----------
    target: array-like of shape [n_samples,]
        The target vector. Should be integers from 0 to n_classes.

    n_classes: int or None
        The total number of classes

    Returns
    -------
    A binary matrix representation of the target.
    """
    target = np.asarray(target, dtype='int').ravel()
    if not n_classes:
        n_classes = np.max(target) + 1

    n_samples = target.shape[0]
    onehot_matrix = np.zeros((n_samples, n_classes), dtype=np.float32)
    onehot_matrix[np.arange(n_samples), target] = 1

    return onehot_matrix

def get_shape(data):
    if isinstance(data, list):
        return len(data), 1
    elif hasattr(data, 'shape'):
        shape = data.shape
        if len(shape) == 1:
            return shape[0], 1
        return shape


class DataBundle(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __repr__(self):
        data_shape = get_shape(self.data)
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


def load_file(file_name, label=0, encoding='utf-8'):
    with open(file_name, 'r') as f:
        texts = [line.decode(encoding).strip() for line in f]

    return pd.DataFrame(dict(data=texts, target=np.repeat(label, len(texts))))


def fetch_dataset(dataset, train_test_split=False, as_onehot=False):
    dataset_train = dataset['train']
    dataset_test = dataset['test']

    x_train = np.asarray(dataset_train.data)
    x_test = np.asarray(dataset_test.data)
    y_train = np.asarray(dataset_train.target)
    y_test = np.asarray(dataset_test.target)
    n_classes = np.max(y_train) + 1

    if as_onehot:
        y_train = to_categorical(y_train, n_classes=n_classes)
        y_test = to_categorical(y_test, n_classes=n_classes)

    if not train_test_split:
        return (np.concatenate((x_train, x_test), axis=0),
                np.concatenate((y_train, y_test), axis=0),
                n_classes)

    return (x_train, y_train), (x_test, y_test), n_classes
