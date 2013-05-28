from matplotlib import pyplot

import numpy as np
from sklearn import cluster

def calc_rows_and_cols(number_of_figures):
    num_cols = num_rows = 1
    if number_of_figures > 1:
        num_cols = 2
        num_rows = (number_of_figures/num_cols) + (1 if number_of_figures % num_cols > 0 else 0)
    return num_rows,num_cols
    
def get_axis(axes,index,number_of_figures,num_rows,num_cols):
    axis = None
    if number_of_figures == 1:
        axis = axes
    elif num_rows == 1:
        axis = axes[index]
    else:
        axis = axes[index/num_cols,index % num_cols]
        
    return axis
    
def plot_clusters(axis,data,k,labels,centroids,alpha=None):
    for i in range(k):
        ds = data[np.where(labels==i)]
        dots = axis.plot(ds[:,0],ds[:,1],'o')
        xs = axis.plot(centroids[i,0],centroids[i,1],'kx')
        pyplot.setp(xs,ms=15.0)
        pyplot.setp(xs,mew=2.0)
        if alpha:
            pyplot.setp(dots,alpha=alpha)
            pyplot.setp(xs,alpha=alpha)
        

def kmeans_predict(data,kmeans,filename=None):
    data = np.array(data)
    
    if type(kmeans) != list:
        kmeans = [kmeans]
        
    number_of_figures = len(kmeans) 
    num_rows,num_cols = calc_rows_and_cols(number_of_figures)
        
    fig, axes = pyplot.subplots(num_rows,num_cols,sharex=True,sharey=True)
    xmin = 1.5
    xmax = 5.5
    ymin = 40
    ymax = 100
    
    index = 0
    for km in kmeans:
        labels = km.predict(data)
        centroids = km.cluster_centers_
        
        axis = get_axis(axes,index,number_of_figures,num_rows,num_cols)
        plot_clusters(axis,data,len(centroids),labels,centroids)
        
        axis.set_title('KMeans Predictions (k=%d)' % len(centroids))
        axis.set_xlabel('Eruption Time (mins)')
        axis.set_ylabel('Time Between Eruptions (mins)')
        axis.set_xlim([xmin,xmax])
        axis.set_ylim([ymin,ymax])
        
        index += 1
        
    
    if filename is not None:
        fig.savefig(filename)
    else:
        pyplot.show()

    

def plot_kmeans(data,ks=(2,),filename=None,suppress_output=False):
    data = np.array(data)
    
    number_of_figures = len(ks)
    num_rows,num_cols = calc_rows_and_cols(number_of_figures)
    
    fig, axes = pyplot.subplots(num_rows,num_cols,sharex=True,sharey=True)
    xmin = 1.5
    xmax = 5.5
    ymin = 40
    ymax = 100
    
    index = 0
    kmeans_models = []
    for k in ks:
        # do k-means clustering
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(data)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        kmeans_models.append(kmeans)
        
        # now plot the clusters
        axis = get_axis(axes,index,number_of_figures,num_rows,num_cols)
        plot_clusters(axis,data,k,labels,centroids)
        
        axis.set_title('Number of Clusters=%d' % k)
        axis.set_xlabel('Eruption Time (mins)')
        axis.set_ylabel('Time Between Eruptions (mins)')
        axis.set_xlim([xmin,xmax])
        axis.set_ylim([ymin,ymax])
        
        index += 1
            
    if suppress_output == False:
        if filename is not None:
            fig.savefig(filename)
        else:
            pyplot.show()

    return kmeans_models
   