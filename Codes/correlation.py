# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import numpy as np
from pandas._libs.algos import nancorr



def nanCorrcoef3D(data):
    '''
    Computes the Pearson correlation coefficients of a 3D array, where the 
    first and third dimentions are the variables and the second dimension is 
    the number of points. Thus, the array has n x m number of variables with p 
    values each. Nan values represent inexistent data and are not used to 
    compute the correlations. The result is a 4D array with the correlations 
    between all possible combinations of the vriables.
    
    Parameters
    ----------
    data : (n,p,m) numpy array
        Total residals.
    
    Returns
    ----------
    corr : (n,n,m,m) numpy array
        Correlation matrix.
    
    '''
    
    n,p,m = data.shape
    
    ordered = np.transpose(data, (0,2,1))
    flat = np.reshape(ordered, (n*m, p))
    corrT = np.reshape(nancorr(flat.T), (n, m, n, m))
    corr = np.transpose(corrT, (0,2,1,3))
    
    return corr



def corrcoef3D(data, axis=1):
    '''
    Computes the Pearson correlation coefficients of a 3D array, where the 
    first and third dimentions are the variables and the second dimension is 
    the number of points. Thus, the array has n x m number of variables with p 
    values each. The result is a 4D array with the correlations 
    between all possible combinations of the vriables.
    
    Parameters
    ----------
    data : (n,p,m) numpy array
        Total residals.
    
    Returns
    ----------
    corr : (n,n,m,m) numpy array
        Correlation matrix.
    
    '''
    
    n,p,m = data.shape
    
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    mean_x = np.tile(mean[:,None,:,None], (1,n,1,m))
    mean_y = np.tile(mean[None,:,None,:], (n,1,m,1))
    std_x = np.tile(std[:,None,:,None], (1,n,1,m))
    std_y = np.tile(std[None,:,None,:], (n,1,m,1))
    mean_xy = np.transpose(np.tensordot(data, data, axes=([1],[1])), (0,2,1,3)) / p
    corr = (mean_xy - mean_x*mean_y) / (std_x*std_y)
    
    return corr




def nanCorrcoef3D_1D(data_3D, data_1D):
    '''
    Computes the Pearson correlation coefficients between a 3D array and a 1D 
    array. The first and third dimentions of the 3D array are the variables and 
    the second dimension is the number of points. Thus, the 3D array has n x m 
    number of variables with p values each. The 1D array represents only one 
    variable with p values. Nan values represent inexistent data and are not 
    used to compute the correlations. The result is a 2D array with the 
    correlations between all variables in the 3D array and the variable of the 
    1D array.
    
    Parameters
    ----------
    data_3D : (n,p,m) numpy array
        Total residals.
    data_1D : (p) numpy array
    
    Returns
    ----------
    corr : (n,m) numpy array
        Correlation matrix.
    
    '''
    
    n,p,m = data_3D.shape
    
    corr = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            corr[i,j] = nancorr(np.vstack([data_3D[i,:,j], data_1D]).T)[0,1]
    
    return corr


