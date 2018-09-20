import numpy as np


def make_gram(numpy_arr: np.ndarray, gram_size: int, stride=1):
    """window sliding method
    x = np.array([1,2,3,4])
    make_gram(x, 2, 1)
    >> array([[1, 2],
       [2, 3],
       [3, 4]])
    make_gram(x, 2, step=2)
    >> array([[1, 2],
       [3, 4]])
    """
    row = len(numpy_arr)
    return np.array([
        numpy_arr[i:i + gram_size]
        for i in range(0, row - gram_size + 1, stride)
    ])


def make_kernel_gram(numpy_arr: np.ndarray,
                     gram_size: int,
                     stride=1,
                     kernel=lambda x: x):
    """ see `n_gram.make_gram`
    `make_kernel_gram` applys a function on each gram.
    """
    row = len(numpy_arr)
    return np.array([
        kernel(numpy_arr[i:i + gram_size])
        for i in range(0, row - gram_size + 1, stride)
    ])
