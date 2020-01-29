import numpy as np
from scipy.optimize.linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo' -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Armijo':
            alpha_0 = previous_alpha if previous_alpha is not None else self.alpha_0
            return self.armijo_search(oracle, x_k, d_k, alpha_0)
        elif self._method == 'Wolfe':
            alpha = scalar_search_wolfe2(
                # phi=lambda step: oracle.func(x_k + step * d_k),
                # derphi=lambda step: oracle.grad(x_k + step * d_k).dot(d_k),
                phi=lambda step: oracle.func_directional(x_k, d_k, step),
                derphi=lambda step: oracle.grad_directional(x_k, d_k, step),
                c1=self.c1,
                c2=self.c2
            )[0]
            if alpha is None:
                return self.armijo_search(oracle, x_k, d_k, self.alpha_0)
            else:
                return alpha

        return None

    def armijo_search(self, oracle, x_k, d_k, alpha_0):
        # phi = lambda step: oracle.func(x_k + step * d_k)
        phi = lambda step: oracle.func_directional(x_k, d_k, step)
        alpha = alpha_0
        # coef = self.c1 * oracle.grad(x_k).dot(d_k)
        coef = self.c1 * oracle.grad_directional(x_k, d_k, 0)
        while phi(alpha) > phi(0) + alpha * coef:
            alpha = alpha / 2
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()
