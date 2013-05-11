import numpy as np
import numpy.linalg as la
import numpy.random as npr

import math

from stats import SingularMatrixError

def dmvnorm(data, mu=None, sigma=None):
    if data is None:
        raise ValueError('No data supplied')    
    num_observations = data.shape[0]
    if num_observations == 0:
        raise ValueError('No data supplied')
    if len(data.shape) != 2:
        raise ValueError('Require 2-D array of data')
    p = data.shape[1]
    
    if mu is None:
        mu = np.zeros(p)
    if sigma is None:
        sigma = np.eye(p)
    
    if mu.shape != (p,):
        raise ValueError('Mean array must be the size of number of variables')
    if sigma.shape != (p,p):
        raise ValueError('Covariance matrix must be square and of size of number of variables')
    
    det = la.det(sigma)
    if det == 0:
        raise SingularMatrixError('Supplied covariance matrix is singular')
    norm_const = 1.0 / (math.pow((2.0*math.pi), p/2.0) * math.pow(det, 1.0/2))
    mean_diff = data - mu
    exponent = np.diagonal(mean_diff.dot(la.inv(sigma)).dot(mean_diff.T))
    
    return norm_const * np.power(math.e, -0.5 * exponent)

def rwish(shape, scale, samples=1):
    if len(scale.shape) != 2:
        if scale.shape[0] != 1:
            raise ValueError('Scale parameter must be a 2-D matrix')
    if scale.shape[0] != scale.shape[1]:
        raise ValueError('Scale parameter must be a square matrix')
    
    p = scale.shape[0]
    if shape < p-1:
        raise ValueError('Shape parameter must be equal to or greater than the number of dimensions.')
    
    chol = la.cholesky(scale)
    result = np.zeros((samples,p,p))
    for i in range(samples):
        z = np.eye(p)
        z = z * np.sqrt(npr.chisquare([x+shape for x in range(0,-p,-1)],p))
        if p > 1:
            pseq = range(1,p)
            z[np.triu_indices(p,1)] = np.random.normal(size=p*(p-1)/2)
        a = chol.dot(z)
        result[i] = a.dot(a.T)
    
    return result

    