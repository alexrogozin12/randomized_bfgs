import numpy as np
import scipy

from methods.line_search import get_line_search_tool
from methods.base import BaseMethod


class Nesterov(BaseMethod):
    def __init__(self, oracle, x_0, stepsize, momentum, tolerance=1e-5,
                 stopping_criteria='grad_rel', trace=True):
        
        super(Nesterov, self).__init__(oracle, x_0, stopping_criteria, trace)
        self.x_k = self.x_0.copy()
        self.y_k = self.x_0.copy()
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        self.stepsize = stepsize
        self.momentum = momentum
        self.tolerance = tolerance

    def step(self):
        x_k_old = self.x_k.copy()
        self.grad_k = self.oracle.grad(self.y_k)
        self.x_k = self.y_k - self.stepsize * self.grad_k
        self.y_k = self.x_k + self.momentum * (self.x_k - x_k_old)

#     def stopping_criteria(self):
#         return np.linalg.norm(self.grad_k)**2 <= self.tolerance * self.grad_norm_0**2
