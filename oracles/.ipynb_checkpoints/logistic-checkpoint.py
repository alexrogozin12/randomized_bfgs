import numpy as np
import scipy
from scipy.special import expit
from oracles import BaseSmoothOracle


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = self.b.shape[0]
        degree1 = np.zeros(m)
        degree2 = self.matvec_Ax(x)
        degree2 = np.multiply(-self.b, degree2)
        summ = np.sum(np.logaddexp(degree1, degree2))

        return summ / m + self.regcoef / 2 * np.dot(x, x)

    def grad(self, x):
        m = self.b.shape[0]
        degrees = -np.multiply(self.b, self.matvec_Ax(x))
        # print('degrees.shape = {}, b.shape={}, m = {}'
        #       .format(degrees.shape, self.b.shape, m))
        sigmas = expit(degrees)
        return -1 / m * self.matvec_ATx(np.multiply(sigmas, self.b)) + self.regcoef * x

    def hess(self, x):
        m = self.b.shape[0]
        n = x.size
        degrees = -np.multiply(self.b, self.matvec_Ax(x))
        sigmas = expit(degrees)
        diagonal = np.multiply(self.b**2, sigmas)
        diagonal = np.multiply(diagonal, 1 - sigmas)
        return np.array(1 / m * self.matmat_ATsA(diagonal) + self.regcoef * np.eye(n))
    
    def hess_mat_prod(self, x, S):
        m = self.b.shape[0]
        n = x.size
        degrees = -np.multiply(self.b, self.matvec_Ax(x))
        sigmas = expit(degrees)
        diagonal = np.multiply(self.b**2, sigmas)
        diagonal = np.multiply(diagonal, 1 - sigmas)
        res = np.multiply(diagonal.reshape(diagonal.shape[0], 1), self.matvec_Ax(S))
        return np.array(1 / m * self.matvec_ATx(res)) + self.regcoef * S


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().transpose() * x

    def matmat_ATsA(s, mat=A):
        if isinstance(mat, np.ndarray):
            return mat.T.dot(np.multiply(mat, s.reshape(len(s), 1)))
        A = mat.tocsr()
        sA = A.multiply(s.reshape(len(s), 1))
        return A.transpose() * sA

    return LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
