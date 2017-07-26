from tqdm import tqdm


def iter_content(response, chunk_size):
    """An iterator over the content of a file download by urlopen."""
    while 1:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        yield [chunk]


def chunk_read(response, chunk_size=8192):
    """Download a file chunk by chunk."""
    total_size = int(response.info()['Content-Length'].strip())
    n_iter = int(total_size / chunk_size)
    data = []

    for datum in tqdm(iter_content(response, chunk_size),
                      total=n_iter, unit='B', unit_scale=True):
        data += datum

    return b"".join(data)
