# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import numpy as np
import pandas as pd
import pickle
from scipy.optimize import least_squares


# Load computed correlations
results = pickle.load(open("correlationResults.pkl", "rb"))
correlations = results['correlations']
correlations_PGV = results['correlations_PGV']
xi = results['xi']
T = results['T']


# Precomputations
nXi = len(xi)
nT = len(T)
Tmin, Tmax = T[0], T[-1]
ind_5p = np.where(xi==0.05)[0][0]
Xi1,Xi2 = np.meshgrid(xi, xi)
xi1 = Xi1.flatten()
xi2 = Xi2.flatten()



# Functional form to be fitted
def quadraticSurf(param, xi1, xi2):
    """
    Functional form for Delta(xi1,xi2) = rho(xi1,xi2) - rho(0.05,0.05).
    """
    
    a,b,c,d,e = param
    
    x = np.log(xi1/0.05)
    y = np.log(xi2/0.05)
    C = a*x**2 + b*y**2 + c*x + d*y +e*x*y
    
    return C

# Special case for the diagonal (same periods)
def quadraticSurf_diag(param, xi1, xi2):
    """
    Functional form for Delta(xi1,xi2) = rho(xi1,xi2) - rho(0.05,0.05).
    """
    
    a = param[0]
    b = a
    c = d = 0
    e = -2*a
    
    x = np.log(xi1/0.05)
    y = np.log(xi2/0.05)
    C = a*x**2 + b*y**2 + c*x + d*y +e*x*y
    
    return C


# Error functions
ErrorFunc = lambda params,xi_1,xi_2,c_val: quadraticSurf(params,xi_1,xi_2) - c_val
ErrorFunc_diag = lambda params,xi_1,xi_2,c_val: quadraticSurf_diag(params,xi_1,xi_2) - c_val



# Perform least squares fitting for each pair of periods
K1 = np.zeros((nT,nT))
K2 = np.zeros((nT,nT))
K3 = np.zeros((nT,nT))
K4 = np.zeros((nT,nT))
K5 = np.zeros((nT,nT))
for i in range(nT):
    for j in range(nT):
    
        RHO = correlations[:,:,i,j]
        rho = RHO.flatten()
        c = rho - RHO[ind_5p,ind_5p]
        
        if i==j:
            param0 = np.array([0.])
            error = ErrorFunc_diag
        else:
            param0 = np.array([0., 0., 0., 0., 0.])
            error = ErrorFunc
        
        result = least_squares(error, param0, args=(xi1,xi2,c), bounds=(-1,1), 
                               method='dogbox')#, method='lm'
        param = result['x']
        
        if i==j:
            a = param[0]
            K1[i,j],K2[i,j],K3[i,j],K4[i,j],K5[i,j] = a,a,0,0,-2*a
        else:
            K1[i,j],K2[i,j],K3[i,j],K4[i,j],K5[i,j] = param



# Save results in CSV files
def saveAsCSV(matrix, filename):
    Tstr = ['T='+str(t) for t in T]
    df = pd.DataFrame(data=matrix, index=Tstr, columns=Tstr)
    df.to_csv(filename)
saveAsCSV(correlations[ind_5p,ind_5p], 'rho5.csv')
saveAsCSV(K1, 'A.csv')
saveAsCSV(K3, 'B.csv')
saveAsCSV(K5, 'C.csv')
