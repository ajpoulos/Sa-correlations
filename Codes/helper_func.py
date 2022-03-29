# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import numpy as np




def faultType(rake):
    '''
    Returns the fault type base on the rake angle.
    
    Parameters
    ----------
    rake : float or numpy array
        Rake angle.
    
    Returns
    ----------
    F : float or numpy array
        1: Strike-slip
        2: Normal
        3: Reverse
    '''
    
    if isinstance(rake, np.ndarray):
        
        F = 2*np.ones(rake.shape, dtype=int)
        F[np.logical_or(np.abs(rake)<=30, (180 - np.abs(rake))<=30)] = 1
        F[(rake > 30)*(rake < 150)] = 3
        
    else:
        
        if np.abs(rake) <= 30 or (180 - np.abs(rake)) <= 30: # strike-slip
            F = 1
        elif 30 < rake < 150: # reverse
            F = 3
        else: # normal
            F = 2
        
    return F




def get_CY_Region(lon, lat):
    """
    Returns the regions of a list of geographical coordinates used for the 
    Chiou and Youngs (2014) ground motion model.
    
    Parameters
    ----------
    lons : (n,) array of floats
        Longitudes.
    lats : (n,) array of floats
        Latitudes.
    
    Returns
    -------
    regions : (n,) array of ints
        Regions:
        0 for Global
        1 for California
        2 for Japan
        3 for China
        4 for Italy
        5 for Turkey
    
    References
    ----------
    1. Chiou, B. S. J., & Youngs, R. R. (2014). Update of the Chiou and Youngs 
       NGA model for the average horizontal component of peak ground motion and 
       response spectra. Earthquake Spectra, 30(3), 1117-1153.
    """
    
    n = len(lon)
    regions = np.zeros(n, dtype=int)
    
    for i in range(n):
        if inCalifornia(lon[i], lat[i]):
            regions[i] = 1
        elif inJapan(lon[i], lat[i]):
            regions[i] = 2
        elif inChina(lon[i], lat[i]):
            regions[i] = 3
        elif inItaly(lon[i], lat[i]):
            regions[i] = 4
        elif inTurkey(lon[i], lat[i]):
            regions[i] = 5
    
    return regions



def inCalifornia(lon, lat):
    return (-126 < lon < -112 and 30 < lat < 43)

def inJapan(lon, lat):
    return (127 < lon < 150 and 29 < lat < 47)

def inChina(lon, lat):
    return (99 < lon < 123 and 26 < lat < 42)

def inItaly(lon, lat):
    return (8 < lon < 17 and 36 < lat < 47)

def inTurkey(lon, lat):
    return (25 < lon < 40 and 36 < lat < 42)



def mixedEffects(epsilon_T, tau, all_phi):
    '''
    Computes the inter- and intra-event residuals of ground motion intensities
    for a given earthquake using a modified version of the method proposed by
    Abrahamson and Youngs (1992) that accounts for phi being different between
    sites.
    
    Parameters
    ----------
    epsilon_T : (n,) numpy array
        Total residals.
    tau : float
        Inter-event standard deviation.
    all_phi : (n,) numpy array
        Intra-event standard deviations.
    
    Returns
    ----------
    eta : floats
        Inter-event residual.
    epsilon : (n,) numpy array
        Intra-event residuals.
    
    References
    ----------
    1. Abrahamson, N. A., & Youngs, R. R. (1992). A stable algorithm for 
       regression analyses using the random effects model. `Bulletin of the
       Seismological Society of America`, 82(1), 505-510.
    '''
    
    
    eta = np.sum(epsilon_T/all_phi**2, axis=0)/(1/tau**2 + 
                                                np.sum(1/all_phi**2, axis=0))
    epsilon = epsilon_T - eta
    
    return eta, epsilon
    
    
    