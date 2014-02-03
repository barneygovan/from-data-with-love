import numpy as np
import numpy.random as npr
import numpy.linalg as la

from matplotlib import pyplot
from faithful.clustering import plot_clusters

import math

from stats.distributions import dmvnorm, rwish

class GaussianFiniteMixtureModel(object):

    def __init__(self, alpha=100.0, g=0.3, delta=1):
        self.__alpha = alpha
        self.__g = g
        self.__delta = delta

    def run(self,data,k,iterations=100):
        data = np.array(data)

        num_observations = data.shape[0]
        p = data.shape[1]

        xi = np.median(data,axis=0)
        dr = np.amax(data,axis=0) - np.amin(data,axis=0)
        kappa = np.zeros((p,p))
        for i in range(p):
            kappa[i,i] = 1.0/(dr[i]**2)
        ikappa = la.inv(kappa)
        h = (100.0*self.__g/self.__alpha) * kappa

        gibbs_pi = np.zeros((iterations,k),dtype=np.double)
        gibbs_theta = np.zeros((iterations,k,p),dtype=np.double)
        gibbs_sigma = np.zeros((iterations,k,p,p),dtype=np.double)
        gibbs_beta = np.zeros((iterations,p,p),dtype=np.double)

        for j in range(k):
            gibbs_pi[0,j] = 1.0/k
            gibbs_theta[0,j] = np.mean(data,axis=0)
            gibbs_sigma[0,j] = np.cov(data.T)

        for i in range(1,iterations):
            a = np.zeros((num_observations,k))
            for m in range(k):
                a[:,m] = gibbs_pi[i-1,m] * dmvnorm(data, mu=gibbs_theta[i-1,m],sigma=gibbs_sigma[i-1,m])
            asum = np.sum(a,axis=1)

            z = np.zeros((num_observations,k),dtype=np.integer)
            for j in range(num_observations):
                z[j] = npr.multinomial(1,a[j]/asum[j])
            gibbs_pi[i] = npr.dirichlet(np.sum(z,axis=0) + self.__delta)

            gibbs_beta[i] = rwish(2.0*self.__g + 2.0*k*self.__alpha,
                                  la.inv(2.0*h + 2.0*self.__sum_isigma(gibbs_sigma[i-1],k,p)))[0]

            y = []
            n = np.zeros(k,dtype=np.integer)
            for m in range(k):
                pos_data = data[np.where(z[:,m]==1)]
                y.append(pos_data)
                n[m] = pos_data.shape[0]

            assert len(y) == k

            for m in range(k):
                y_diff = y[m] - gibbs_theta[i-1,m]

                y_sum = np.zeros((p,p),dtype=np.double)
                for l in range(n[m]):
                    y_sum = y_sum + y_diff[l].reshape((p,1)) * y_diff[l]

                cov = (2.0 * gibbs_beta[i]) + y_sum

                gibbs_sigma[i,m] = la.inv(rwish(2.0*self.__alpha + n[m], la.inv(cov))[0])

            for m in range(k):
                cov = la.inv(n[m] * la.inv(gibbs_sigma[i,m]) + kappa)
                mean_x = np.mean(y[m],axis=0)
                mean = cov.dot(n[m] * la.inv(gibbs_sigma[i,m]).dot(mean_x) + kappa.dot(xi))
                gibbs_theta[i,m] = npr.multivariate_normal(mean,cov,size=1)

        return (gibbs_pi,gibbs_theta,gibbs_sigma)

    def __sum_isigma(self, sigma, k, p):
        tmp = np.zeros((p,p),dtype=np.double)
        for i in range(k):
            tmp = tmp + la.inv(sigma[i])
        return tmp

def calculate_posterior_means(pi, mu, sigma, burnin):
    pi_mean = np.mean(pi[burnin:,], axis=0)
    mu_mean = np.mean(mu[burnin:,], axis=0)
    mu_sigma = np.mean(sigma[burnin:,], axis=0)
    return (pi_mean, mu_mean, mu_sigma)

def __gaussian_mixture(x, y, pi, mu, sigma):
    z = np.zeros((len(x), len(y)))
    number_of_clusters = pi.shape[0]
    p = mu.shape[1]
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):
            data = np.array([xval, yval]).reshape((1,p))
            zval = 0
            for k in range(number_of_clusters):
                zval += pi[k] * dmvnorm(data, mu=mu[k], sigma=sigma[k])
            z[i,j] = zval
    return z

def draw_contour_plots(pi, mu, sigma, filename=None):
    fig, axis = pyplot.subplots(1,1)
    number_of_steps = 100.0
    xmin = 1.5
    xmax = 5.5
    xstep = (xmax-xmin)/number_of_steps
    ymin = 40
    ymax = 100
    ystep = (ymax-ymin)/number_of_steps

    x = np.arange(xmin, xmax, xstep)
    y = np.arange(ymin, ymax, ystep)
    z = __gaussian_mixture(x, y, pi, mu, sigma)

    c = axis.contourf(x, y, z, 120)
    cb = pyplot.colorbar(c)

    if filename:
        fig.savefig(filename)
    else:
        pyplot.show()

