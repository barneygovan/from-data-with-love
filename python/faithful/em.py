
import numpy as np
import numpy.random as npr
import numpy.linalg as la

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from stats.distributions import dmvnorm
from faithful.clustering import plot_clusters

def plot_against_kmeans(data,theta_path,kmeans,filename=None,suppress_output=False):
    data = np.array(data)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    k = len(centroids)
    
    fig,axis = pyplot.subplots(1,1)
    plot_clusters(axis,data,k,labels,centroids,alpha=0.5)
    xmin = 1.5
    xmax = 5.5
    ymin = 40
    ymax = 100
    
    axis.set_title('EM model vs. KMeans.  \nNumber of Iterations: %d' % len(theta_path))
    axis.set_xlabel('Eruption Time (mins)')
    axis.set_ylabel('Time Between Eruptions (mins)')
    axis.set_xlim([xmin,xmax])
    axis.set_ylim([ymin,ymax])
    
    lines = axis.plot(theta_path[:,0,0],theta_path[:,0,1],'g-',theta_path[:,1,0],theta_path[:,1,1],'b-')
    pyplot.setp(lines,lw=2.0)
    xs = axis.plot(theta_path[-1,0,0],theta_path[-1,0,1],'rx',theta_path[-1,1,0],theta_path[-1,1,1],'rx')
    pyplot.setp(xs,ms=15.0)
    pyplot.setp(xs,mew=2.0)
    
    
    if suppress_output == False:
        if filename is not None:
            fig.savefig(filename)
        else:
            pyplot.show()
    

def gaussian_mixture(x,y,theta,sigma,pi):
    
    z = np.zeros((len(x),len(y)))
    
    for i,xval in enumerate(x):
        for j,yval in enumerate(y):
            data = np.array([xval,yval]).reshape((1,2))
            z1 = dmvnorm(data,mu=theta[0],sigma=sigma[0])
            z2 = dmvnorm(data,mu=theta[1],sigma=sigma[1])
            z[j,i] = ((1-pi) * z1) + (pi * z2)
    
    return z
    
def draw_contour_plots(theta,sigma,pi,filename=None):
    
    #fig = pyplot.figure()
    #axis = fig.add_subplot(111,projection='3d')
    fig,axis = pyplot.subplots(1,1)
    number_of_steps = 100.0
    xmin = 1.5
    xmax = 5.5
    xstep = (xmax-xmin)/number_of_steps
    ymin = 40
    ymax = 100
    ystep = (ymax-ymin)/number_of_steps
    
    x = np.arange(xmin,xmax,xstep)
    y = np.arange(ymin,ymax,ystep)
    z = gaussian_mixture(x,y,theta,sigma,pi)
    #x,y = np.meshgrid(x,y)
    
    #c = axis.plot_surface(x,y,z)
    c = axis.contourf(x,y,z,120)
    cb = pyplot.colorbar(c)
    #pyplot.clabel(c, inline=1, fontsize=10)
    
    if filename is not None:
        fig.savefig(filename)
    else:
        pyplot.show()
    
def gaussian_em_2(data,max_reps=10,init_theta=None,init_sigma=None,init_pi=None,delta=0.001):
    data = np.array(data)
    
    theta = init_theta
    sigma = init_sigma
    pi = init_pi
    
    theta_history = []
    sigma_history = []
    pi_history = []
    
    if theta is None:
        theta = np.zeros((2,2))
        theta[0] = data[npr.randint(0,len(data))]
        theta[1] = data[npr.randint(0,len(data))]
    if sigma is None:
        sigma = np.zeros((2,2,2))
        sigma[0] = np.cov(data,rowvar=0)
        sigma[1] = np.copy(sigma[0])
    if pi is None:
        pi = 0.5
    
    
    theta_history.append(theta)
    sigma_history.append(sigma)
    pi_history.append(pi)
    
    for r in range(max_reps):
        grouping,responsibilities = _expectation_gauss2(data,theta,sigma,pi)
        theta,sigma,pi = _maximization_gauss2(data,responsibilities)
        theta_history.append(theta)
        sigma_history.append(sigma)
        pi_history.append(pi)
        if _is_converged(theta_history,sigma_history,pi_history,delta):
            break
        
    return np.array(theta_history),np.array(sigma_history),np.array(pi_history)

def _expectation_gauss2(data,theta,sigma,pi):
    
    prob_cluster1 = (1-pi) * dmvnorm(data,mu=theta[0],sigma=sigma[0])
    prob_cluster2 = pi * dmvnorm(data,mu=theta[1],sigma=sigma[1])
    
    grouping = np.zeros_like(prob_cluster1,dtype=np.integer)
    grouping[np.where(prob_cluster1<prob_cluster2)] = 1
    
    responsibilities = prob_cluster2 / ( prob_cluster1 + prob_cluster2)
    
    return grouping,responsibilities

    
def _maximization_gauss2(data,responsibilities):
    theta_1 = np.sum( (1-responsibilities) * data.T, axis=1) / np.sum( 1-responsibilities )
    theta_2 = np.sum( responsibilities * data.T, axis=1) / np.sum(responsibilities)
    
    sigma_1 = ((data - theta_1) * (1-responsibilities)[:,np.newaxis]).T.dot(data - theta_1) / np.sum(1-responsibilities)
    sigma_2 = ((data - theta_2) * responsibilities[:,np.newaxis]).T.dot(data - theta_2) / np.sum(responsibilities)
    
    pi = np.sum(responsibilities)/len(responsibilities)
    
    return np.append(theta_1,theta_2).reshape((2,2)), np.append(sigma_1,sigma_2).reshape((2,2,2)), pi
    
def _is_converged(theta_history,sigma_history,pi_history,delta):
    if len(theta_history) < 2:
        return False
        
    if la.norm(theta_history[-1]-theta_history[-2]) > delta:
        return False
        
    last_sigma = sigma_history[-2]
    present_sigma = sigma_history[-1]
    if la.norm(present_sigma - last_sigma) > delta:
        return False
        
    last_pi = pi_history[-2]
    present_pi = pi_history[-1]
    if present_pi - last_pi > delta:
        return False
        
    return True
    
    
