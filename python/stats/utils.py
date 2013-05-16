import numpy as np
import numpy.random as npr

def split_data(data,train_split=0.8):
    data = np.array(data)
    num_train = data.shape[0] * train_split
    npr.shuffle(data)
    
    return (data[:num_train],data[num_train:])
    