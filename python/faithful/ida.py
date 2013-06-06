from matplotlib import pyplot
from scipy import stats

import numpy as np
import numpy.linalg as la

def scatter_plot(data,filename=None):
    data = np.array(data)
    fig, axis = pyplot.subplots(1,1)
   
    axis.plot(data[:,0],data[:,1],'ro')
    axis.set_xlabel('Eruption Time (minutes)')
    axis.set_ylabel('Time Between Eruptions (minutes)')
    axis.set_title('Old Faithful Eruptions')
    
    if filename is not None:
        fig.savefig(filename)
    else:
        pyplot.show()
        
def _set_axis_properties(axis_eruptions,axis_waiting):
    axis_eruptions.set_xlim(0,6)
    axis_eruptions.set_xlabel('Eruption Time (minutes)')
    axis_eruptions.set_ylabel('Count')
    
    axis_waiting.set_xlim(40,100)
    axis_waiting.set_xlabel('Time Between Eruptions (minutes)')
    axis_waiting.set_ylabel('Count')
    
def histogram(data,filename=None):
    data = np.array(data)
    fig, (axis_eruptions,axis_waiting) = pyplot.subplots(1,2,sharex=False,sharey=False)
    
    _set_axis_properties(axis_eruptions,axis_waiting)
    axis_eruptions.hist(data[:,0])
    axis_waiting.hist(data[:,1])
    
    add_kernel_density_estimate(data,(fig,axis_eruptions,axis_waiting))
    
    if filename is not None:
        fig.savefig(filename)
    else:
        pyplot.show()
        
    
def add_kernel_density_estimate(data, graph=None, filename=None):
    data = np.array(data)
    fig,axis_eruptions,axis_waiting = None,None,None
    if graph is None:
        fig, (axis_eruptions,axis_waiting) = pyplot.subplots(1,2,sharex=False,sharey=False)
        _set_axis_properties(axis_eruptions,axis_waiting)
    else:
        fig,axis_eruptions,axis_waiting = graph
        axis_eruptions = axis_eruptions.twinx()
        axis_waiting = axis_waiting.twinx()
        
    fig.subplots_adjust(wspace=0.5)
    axis_eruptions.set_ylabel('Density')
    axis_waiting.set_ylabel('Density')
    
    density_eruptions = stats.gaussian_kde(data[:,0])
    density_waiting = stats.gaussian_kde(data[:,1])
    
    x_eruptions = np.arange(0.,6,0.05)
    axis_eruptions.plot(x_eruptions, density_eruptions(x_eruptions),'k-')
    
    x_waiting = np.arange(40.,100.,0.5)
    axis_waiting.plot(x_waiting, density_waiting(x_waiting), 'k-')
    
    if filename is not None:
        fig.savefig(filename)
        
    
def linear_regression(data,filename=None):
    data = np.array(data)
    
    x = data[:,0]
    y = data[:,1]
    X = np.array( [ x, np.ones(len(x)) ] )
    
    slope,intercept = la.lstsq(X.T, y)[0]
    regression = intercept + slope*x
    
    correlation_coeff = np.corrcoef(x,y)[0,1]
    
    fig, axis = pyplot.subplots(1,1)
    axis.plot(x,regression,'b-',x,y,'ro')
    axis.set_xlabel('Eruption Time (minutes)')
    axis.set_ylabel('Time Between Eruptions (minutes)')
    axis.set_title('Old Faithful Eruptions')
    
    box_props = dict(boxstyle='round', fc='w', ec='0.5', alpha=0.9)
    axis.text(1.75,95, 'Slope: %f \nIntercept: %f \nCorrelation: %f' % (slope,intercept,correlation_coeff), va='top', size=10, bbox=box_props)
    
    if filename is not None:
        fig.savefig(filename)
    else:
        pyplot.show()
        
