import numpy as np


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
    """Determine the shape of a dataset."""
    if isinstance(data, list):
        return len(data), 1
    elif hasattr(data, 'shape'):
        shape = data.shape
        if len(shape) == 1:
            return shape[0], 1
        return shape
