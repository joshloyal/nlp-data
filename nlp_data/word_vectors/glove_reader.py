from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import collections
import io
import logging
import mmap
import multiprocessing
import os

from joblib import Parallel, delayed

import numpy as np


debug = False

if debug:
    mp_logger = multiprocessing.log_to_stderr()
    mp_logger.setLevel(logging.INFO)


def get_chunks(filename, chunk_size=1024*1024):
    """Breaks a file into equal size chunks of memory.

    Parameters
    ----------
    filename : str
        Name of the file on disk.
    chunk_size : int
        Size in bytes for each chunk.
    """
    with open(filename, 'r') as savefile:
        while True:
            start = savefile.tell()
            savefile.seek(chunk_size, io.SEEK_CUR)
            line = savefile.readline()
            yield start, savefile.tell() - start
            if not line:
                break


def chunk_it(filename, chunk_size=10*1024*1024, n_groups=1):
    """Generates equal groups of chunks of a file for splitting amoung processes.

    Parameters
    ----------
    filename : str
        Name of the file on disk.
    chunk_size : int
        Size in bytes for each chunk.
    n_groups : int
        Number of equal size groups.
    """
    if n_groups == 1:
        yield get_chunks(filename, chunk_size=chunk_size)
    else:
        chunks = [chunk for chunk in get_chunks(filename, chunk_size=chunk_size)]
        n_chunks = len(chunks)
        chunks_per_group = int(n_chunks / n_groups)
        if n_groups * chunks_per_group < n_chunks:
            chunks_per_group += 1 # add one to all chunks to redistribute left-overs
        for i in xrange(0, len(chunks), chunks_per_group):
            yield chunks[i:i+chunks_per_group]


def stanford_worker(file_handle, chunks, out_q):
    """The main worker process responsible for reading the stanford word vector files.
    The result of this process is a dictionary mapping words to word vectors stored
    as numpy arrays.

    Parameters
    ----------
    file_handle : file object or mmap
        File handle of the stanford pre-trained word vector file
    chunks : tuple
        Starting and ending chunks to process
    out_q : multiprocessing.Queue
        Queue to add the resulting dictionary to
    """
    vocab = {}
    for chunk in chunks:
        file_handle.seek(chunk[0])
        for line in file_handle.read(chunk[1]).splitlines():
            tokens = line.decode('utf-8').strip().split(' ')

            word = tokens[0]
            entries = tokens[1:]

            vocab[tokens[0]] = np.array([float(x) for x in entries])

    if debug:
        mp_logger.info('DONE!')

    out_q.put(vocab)


def from_stanford(filename, n_jobs=4, chunk_size=1024*1024):
    """Parallel version of the from_stanford function.
    This is roughly a 3x speed-up from the serial version.
    """

    # use a multiprocessing Queue to syncronize all of the sub-vocabulary dictionaries
    out_q = multiprocessing.Queue()
    procs = []

    # NOTE: using seek in python 3 is broken, so find out how to fix this
    with open(filename) as glove_file:
        # memory map the file to so that we don't need to copy the file to each sub-process.
        # I'm not sure if this is necessary....
        # we may want to use regex's here as well when searching for word vectors?
        filemap = mmap.mmap(
            glove_file.fileno(),
            os.path.getsize(filename),
            access=mmap.ACCESS_READ)

        n_procs = 0  # we try and use n_jobs, but this may be too much
        for chunk in chunk_it(filename, n_groups=n_jobs, chunk_size=chunk_size):
            p = multiprocessing.Process(
                target=stanford_worker,
                args=(filemap, chunk, out_q))
            procs.append(p)
            p.start()
            n_procs += 1

        # Collect all the results into a single dict. We know how many dicts with results
        # to expect. We use an OrderedDict to avoid having to re-calculate the word_ids;
        # however, this means that successive runs will have different word_ids which
        # will be an un-expected behavior for most users.
        vocabulary = collections.OrderedDict()
        for i in range(n_procs):
            vocabulary.update(out_q.get())

        # Wait for all worker processes to finish before calling join
        for p in procs:
            p.join()

        return vocabulary
