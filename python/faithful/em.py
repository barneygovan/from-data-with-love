
import numpy as np
import numpy.random as npr

from matplotlib import pyplot

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
    
    axis.set_title('EM model vs. KMeans')
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
    

def gaussian_em_2(data,reps=10,init_theta=None,init_sigma=None,init_pi=None):
    data = np.array(data)
    
    theta = init_theta
    sigma = init_sigma
    pi = init_pi
    
    theta_history = np.zeros((reps+1,2,2))
    sigma_history = np.zeros((reps+1,2,2,2))
    pi_history = np.zeros(reps+1)
    
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
    
    
    theta_history[0] = theta
    sigma_history[0] = sigma
    pi_history[0] = pi
    
    for r in range(reps):
        grouping,responsibilities = _expectation_gauss2(data,theta,sigma,pi)
        theta,sigma,pi = _maximization_gauss2(data,responsibilities)
        theta_history[r+1] = theta
        sigma_history[r+1] = sigma
        pi_history[r+1] = pi
        
    return theta_history,sigma_history,pi_history

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
