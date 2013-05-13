from matplotlib import pyplot
    
def stationarity_plot(data,ylab=None,filename=None):
    
    number_of_figures = 1
    
    dims = data.shape
    num_dims = len(dims)
    
    if num_dims > 4:
        raise ValueError('Can only take 1-, 2-, or 3-D array for data')
    
    for d in range(num_dims-1,0,-1):
        number_of_figures *= data.shape[d]
        
    num_cols = num_rows = 1
    if number_of_figures > 1:
        num_cols = 2
        num_rows = (number_of_figures/num_cols) + (1 if number_of_figures % num_cols > 0 else 0)
    
    fig, axes = pyplot.subplots(num_rows,num_cols, sharex=True, sharey=False)
    for i in range(number_of_figures):
        data_i = data
        if num_dims == 2:
            data_i = data[:,i]
        if num_dims == 3:
            data_i = data[:,i/dims[1],i%dims[1]]
        if num_dims == 4:
            data_i = data[:, i%dims[1], (i/dims[1])%dims[2], i/(dims[1] * dims[2])]
        
        s = len(data_i)
        
        axis =  None
        if number_of_figures == 1:
            axis = axes
        elif num_rows == 1:
            axis = axes[i]
        else:
            axis = axes[i/num_cols,i % num_cols]
        
        step_size = 10
        ng = s/step_size
        while ng > 10:
            step_size *= 10
            ng = s/step_size
        bp_data = []
        scan = range(0, s, s/ng)
        scan.append(s)
        for j in range(len(scan) - 1):
            bp_data.append(data_i[scan[j]:scan[j+1]])
            
        axis.boxplot(bp_data,positions=scan[1:],widths=step_size/2.0)
        axis.set_xlim((0,scan[-1] + step_size))
        if i >= (number_of_figures - num_cols):
            axis.set_xlabel('Iteration Number')
        if ylab is not None:
            ylabel = '$%s_{%s}$' % (ylab, str(i))
            axis.set_ylabel(ylabel)
                
        
    if filename is not None:
        fig.savefig(filename)
    else:
        pyplot.show()

