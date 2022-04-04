import numpy as np

def variance(matrix):
    vari  = []
    matrix =  np.array(matrix)
    for i in range(10):
        data = matrix[:,i]
        vari.append(np.var(data))
    sum_var = np.sum(vari)
    return sum_var
