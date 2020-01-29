import numpy as np
import scipy
import pickle

import sys
sys.path.append('../')

from methods import RBFGS
from methods.rbfgs import Uniform, Gaussian, CustomDiscrete


def select_basis_columns(mat, tolerance=1e-10, return_copy=False):
    '''
    Performs QR-decomposition of mat and then selects columns 
    corresponding to entries of R diagonal greater than tolerance
    in absolute value
    '''
    Q, R = scipy.linalg.qr(mat)
    if return_copy:
        return mat[:, np.abs(R.diagonal()) > 1e-10].copy()
    else:
        return mat[:, np.abs(R.diagonal()) > 1e-10]


def read_results_from_files(filenames):
    import pickle
    res = []
    for filename in filenames:
        with open(filename, 'rb') as file:
            res.append(pickle.load(file))
    return res
