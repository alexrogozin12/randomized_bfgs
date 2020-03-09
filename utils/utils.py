import numpy as np
import scipy
import pickle

import sys
sys.path.append('../')

from methods import RBFGS
from methods.rbfgs import Uniform, Gaussian, CustomDiscrete


def read_results_from_files(filenames):
    import pickle
    res = []
    for filename in filenames:
        with open(filename, 'rb') as file:
            res.append(pickle.load(file))
    return res
