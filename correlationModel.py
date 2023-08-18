# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import numpy as np
import pandas as pd
from scipy import interpolate



def correlationModel(T1, xi1, T2, xi2):
    '''
    Computes the correlation between spectral accelerations of different 
    damping ratios and periods using the model developed by Poulos & Miranda 
    (2023).
    
    Parameters
    ----------
    T1 : float
        First period of vibration.
    xi1 : float
        First damping ratio.
    T2 : float
        Second period of vibration.
    xi2 : float
        Second damping ratio.
    
    Returns
    ----------
    rho : float
        Correlation.
    
    References
    ----------
    .. [1] Poulos, A., and Miranda, E. (2023). Damping-dependent correlations 
        between response spectral ordinates. `Earthquake Engineering & 
        Structural Dynamics`, 52(4), 1078-1090.
    '''
    
    # params = [f_A(T2,T1), f_A(T1,T2), f_B(T2,T1), f_B(T1,T2), f_C(T2,T1)]
    params = [f_A(T1,T2), f_A(T2,T1), f_B(T1,T2), f_B(T2,T1), f_C(T1,T2)]
    
    rho = f_rho_5(T1,T2) + _quadraticSurf(params, xi1, xi2)
    
    if len(rho)==1:
        rho = rho[0]
    
    return rho




def _readCSV(filename):
    '''
    Reads a CSV file that was saved by the least squares fitting.
    
    Parameters
    ----------
    filename : string
        Name of the CSV file.
    
    Returns
    ----------
    data : (n,n) numpy array
        Data matrix.
    periods : (n,) numpy array
        Periods of the columns and rows.
    '''
    
    df = pd.read_csv(filename, index_col=0)
    data = df.values
    
    columnNames = np.array(df.columns)
    periods = np.array([float(name[2:]) for name in columnNames])
    
    return data, periods

rho_5, T = _readCSV('rho5.csv')
A, __ = _readCSV('A.csv')
B, __ = _readCSV('B.csv')
C, __ = _readCSV('C.csv')

# Interpolate parameters
f_rho_5 = interpolate.interp2d(T, T, rho_5)
f_A = interpolate.interp2d(T, T, A)
f_B = interpolate.interp2d(T, T, B)
f_C = interpolate.interp2d(T, T, C)
    


def _quadraticSurf(param, xi1, xi2):
    """
    Functional form for Delta = rho(xi1,xi2) - rho(0.05,0.05).
    """
    
    a,b,c,d,e = param
    
    x = np.log(xi1/0.05)
    y = np.log(xi2/0.05)
    Delta = a*x**2 + b*y**2 + c*x + d*y +e*x*y
    
    return Delta

