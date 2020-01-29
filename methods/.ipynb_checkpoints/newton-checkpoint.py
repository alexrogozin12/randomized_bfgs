import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from methods.line_search import get_line_search_tool
from methods.base import BaseMethod


class Newton(BaseMethod):
    def __init__(self, oracle, x_0, tolerance=1e-5, line_search_options=None, 
                 stopping_criteria='grad_rel', trace=True):
        
        super(Newton, self).__init__(oracle, x_0, stopping_criteria, trace)
        self.x_k = self.x_0.copy()
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        if self.line_search_tool._method == 'Constant':
            self.alpha_0 = self.line_search_tool.c
        else:
            self.alpha_0 = 1.

        try:
            self.alpha_k = self.line_search_tool.alpha_0
        except AttributeError:
            self.alpha_k = self.line_search_tool.c
        
    def step(self):
        self.grad_k = self.oracle.grad(self.x_k)
        self.hess_k = self.oracle.hess(self.x_k)

        try:
            factor = scipy.linalg.cho_factor(self.hess_k)
            d_k = scipy.linalg.cho_solve(factor, self.grad_k)
        except LinAlgError:
            return

        self.alpha_k = self.line_search_tool.line_search(
            self.oracle, self.x_k, -d_k, self.alpha_0)
        if self.alpha_k is None:
            return
        
        self.x_k = self.x_k - self.alpha_k * d_k
    
#     def stopping_criteria(self):
#         return np.linalg.norm(self.grad_k)**2 <= self.tolerance * self.grad_norm_0**2
