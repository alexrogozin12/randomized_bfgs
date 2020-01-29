import numpy as np
import scipy
from collections import defaultdict
from datetime import datetime
from numpy.linalg import LinAlgError
from methods.base import BaseMethod
from methods.line_search import get_line_search_tool


class RBFGS(BaseMethod):
    def __init__(self, oracle, x_0, s_distr, tolerance=1e-10, B_0=None, 
                 line_search_options=None, stopping_criteria='grad_rel', 
                 trace=True):
        
        super(RBFGS, self).__init__(oracle, x_0, stopping_criteria, trace)
        self.s_distr = s_distr
        self.d = x_0.shape[0]
        if B_0 is not None:
            self.B_0 = B_0.copy()
        else:
            self.B_0 = np.eye(self.d)
        self.B_k = self.B_0.copy()
        self.x_0 = x_0.copy()
        self.x_k = x_0.copy()
        self.id_mat = np.eye(self.d)
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        self.tolerance = tolerance

    def step(self):
        self.grad_k = self.oracle.grad(self.x_k)
        
        S = self.s_distr.sample()
        HS = self.oracle.hess_mat_prod(self.x_k, S)
        STHS_inv = scipy.linalg.inv(S.T.dot(HS))
        G = S.dot(STHS_inv)
        GH = G.dot(HS.T)
        G = G.dot(S.T)
        self.B_k = self.B_k - GH.dot(self.B_k) # Left multiply by (I - GH)
        self.B_k = self.B_k - self.B_k.dot(GH.T) # Right multiply by (I - HG)
        self.B_k = G + self.B_k
        
        d_k = -self.B_k.dot(self.grad_k)
        alpha_k = self.line_search_tool.line_search(self.oracle, self.x_k, d_k)
        x_k_new = self.x_k + alpha_k * d_k
        
        try:
            last_func_val = self.hist['func'][-1]
        except (IndexError, AttributeError):
            last_func_val = self.oracle.func(self.x_k)
        if self.oracle.func(x_k_new) < last_func_val:
            self.x_k = x_k_new

#     def stopping_criteria(self):
#         return np.linalg.norm(self.grad_k)**2 <= self.tolerance * self.grad_norm_0**2

    def update_B(self):
        S = self.s_distr.sample()
        HS = self.oracle.hess_mat_prod(self.x_k, S)
        STHS_inv = scipy.linalg.inv(S.T.dot(HS))
        G = S.dot(STHS_inv)
        GH = G.dot(HS.T)
        G = G.dot(S.T)
        self.B_k = self.B_k - GH.dot(self.B_k) # Left multiply by (I - GH)
        self.B_k = self.B_k - self.B_k.dot(GH.T) # Right multiply by (I - HG)
        self.B_k = G + self.B_k


class MatrixDistribution(object):
    def __init__(self):
        pass
    
    def sample(self):
        raise NotImplementedError('sample not implemented')


class Uniform(MatrixDistribution):
    def __init__(self, low, high, size):
        super(Uniform, self).__init__()
        self.low = low
        self.high = high
        self.size = size
    
    def sample(self):
        return np.random.uniform(self.low, self.high, self.size)


class Gaussian(MatrixDistribution):
    def __init__(self, mean, std, size):
        super(Gaussian, self).__init__()
        self.mean = mean
        self.std = std
        self.size = size
    
    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.std, size=self.size)


class Identity(MatrixDistribution):
    def __init__(self, d):
        super(Identity, self).__init__()
        self.mat = np.eye(d)
    
    def sample(self):
        return self.mat


class CustomDiscrete(MatrixDistribution):
    '''
    Sample a set of random columns from given matrix.
    '''
    def __init__(self, mat, probs=None, size=None):
        super(CustomDiscrete, self).__init__()
        self.mat = mat.copy()
        self.probs = probs or np.ones(mat.shape[1]) / mat.shape[1]
        if probs is not None and probs.sum() != 1.:
            raise ValueError('Probabilities should sum to 1')
        self.size = size or 1
        self.ids = np.arange(0, mat.shape[1])
    
    def sample(self):
        return self.mat[:, np.random.choice(self.ids, replace=False, 
                                            p=self.probs, size=self.size)]
