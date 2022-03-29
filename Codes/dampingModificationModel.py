# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import numpy as np
import pandas as pd
import inspect, os


rootPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))



df = pd.read_csv(os.path.join(rootPath, 'coefficients.csv'))
values = df.values
__Periods = values[:,0]
__Table = values[:,1:]


df2 = pd.read_csv(os.path.join(rootPath, 'rho_DSF.csv'))
values = df2.values
__T_rho = values[:,0]
__rho = values[:,1:]
__xi_rho = np.array(list(map(float, list(df2)[1:])))



def Rezaeian2014(beta, M, Rrup, T):
    """
    Computes damping scaling factors (DSF).
    
    Parameters
    ----------
        beta : float
            Critical damping ration in percentage (e.g., use 5 for 5%).
        M : float
            Earthquake magnitude.
        Rrup : float
            Rupture distance.
        T : (n,) numpy array
            Structural periods (s).
    Returns
    ----------
        mu : (n,) numpy array
            Mean of log(DSF).
        sigma : (n,) numpy array
            Standard deviation of log(DSF).
        rho : (n,) numpy array
            Correlation between log(DSF) and log(PSA_5%).
    References
    ----------
    1. Rezaeian, S., Bozorgnia, Y., Idriss, I. M., Abrahamson, N., Campbell, 
       K., & Silva, W. (2014). Damping scaling factors for elastic response 
       spectra for shallow crustal earthquakes in active tectonic regions:
       “Average” horizontal component. `Earthquake Spectra`, 30(2), 939-963.
    """
    
    b0,b1,b2,b3,b4,b5,b6,b7,b8,a0,a1 = __interp_2_vec(T, __Periods, __Table)
    
    lb = np.log(beta)
    lb5 = np.log(beta/5)
    
    mu = b0 + b1*lb + b2*lb**2 + (b3 + b4*lb + b5*lb**2)*M + (b6 + b7*lb + b8*lb**2)*np.log(Rrup + 1)
    
    sigma = np.abs(a0*lb5 + a1*lb5**2)
    
    ind = np.argmin(np.abs(__xi_rho-beta))
    rho = np.interp(T, __T_rho, __rho[:,ind])
    
    return mu, sigma, rho


def __interp_2(z, x, y) :
    rows, cols = y.shape
    row_idx = np.arange(cols)#.reshape((rows,) + (1,))
    col_idx = np.argmax(x > z) - 1
    ret = y[col_idx + 1, row_idx] - y[col_idx, row_idx]
    ret /= x[col_idx + 1] - x[col_idx]
    ret *= z - x[col_idx]
    ret += y[col_idx, row_idx]
    return ret

def __interp_2_vec(z, x, y) :
    rows, cols = y.shape
    col_idx = np.argmax(np.greater.outer(x, z), axis=0) - 1
    ret = (y[col_idx + 1, :] - y[col_idx, :]).T
    ret /= x[col_idx + 1] - x[col_idx]
    ret *= z - x[col_idx]
    ret += y[col_idx, :].T
    return ret

