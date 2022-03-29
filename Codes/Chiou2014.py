# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 01:25:37 2019

@author: alanp
"""

import os
import inspect
import numpy as np
import pandas as pd


# Reads a CSV file with the coefficients of the model. This file must be
# located in the same folder as the code.
rootPath = os.path.dirname(os.path.abspath(inspect.getfile(
                                            inspect.currentframe())))
df = pd.read_csv(os.path.join(rootPath, 'Chiou_and_Youngs_2014.csv'))
__Chiou_2014_Coeff = df[['c1','c1a','c1b','c1c','c1d','cn','cM','c3','c5',
                         'cHM','c6','c7','c7b','c8','c8b','c9','c9a','c9b',
                         'c11b','cg1','cg2','cg3','phi1','phi2','phi3','phi4',
                         'phi5','phi6','gammaJp_It','gammaWn','phi1Jp',
                         'phi5Jp','phi6Jp','tau1','tau2','sigma1','sigma2',
                         'sigma3','sigma2Jp']].values
__Chiou_2014_Periods = df['Period (s)'].values




def Chiou_2014(M, Rjb, Rrup, Rx, Ztor, T, Vs30, F=0, dip=90, z1=-999, region=0,
               F_Vs30=1, F_HW=None):
    '''
    Ground motion model developed by Chiou and Youngs (2014) using the
    NGA-West2 database.
    
    Parameters
    ----------
    M : floats
        Moment magnitude.
    Rjb : float
        Joyner-Boore distance (km).
    Rrup : float
        Rupture distance (km).
    Rx : float
        Horizontal distance from top of rupture measured perpendicular to fault
        strike (km)
    Ztor : float
        Depth to the top of ruptured plane (km)
    T : 1-D sequence of floats
        Structural periods (s). Use np.array([-1]) to get PGV.
    Vs30 : float
        Average shear wave velocity of the upper 30 m (in m/s).
    F : int
        Fault type:
        0 for unspecified fault,
        1 for strike-slip,
        2 for normal,
        3 for reverse
    dip : float
        Fault dip angle in degrees.
    z1 : float
        Sediment thickness (m): depth to the shear wave velocity horizon of
        1 km/s. Use -999 if unknown.
    region : int
        0 for Global,
        1 for California,
        2 for Japan,
        3 for China,
        4 for Italy,
        5 for Turkey
    F_Vs30 : int
        0 if Vs30 was inferred from geology,
        1 if Vs30 was measured
    F_HW : int
        0 if site is not in the hanging wall,
        1 if site is in the hanging wall,
        None to computes F_HW as 1*(Rx>=0)
    
    Returns
    ----------
    ln_y : 1-D sequence of floats
        Mean natural logarithm of the ground motion intensity. Spectral
        accelerations are in units of g and PGV is in cm/s.
    sigma : 1-D sequence of floats
        Total logarithmic standard deviation.
    intra : 1-D sequence of floats
        Intra-event (withi-event) logarithmic standard deviation.
    inter : 1-D sequence of floats
        Inter-event (between-event) logarithmic standard deviation.
    
    References
    ----------
    1. Chiou, B. S. J., & Youngs, R. R. (2014). Update of the Chiou and Youngs
       NGA model for the average horizontal component of peak groundmotion and
       response spectra. `Earthquake Spectra`, 30(3), 1117-1153.
    '''
    
    if F==3:
        F_RV = 1
    else:
        F_RV = 0
    if F==2:
        F_NM = 1
    else:
        F_NM = 0
    
    if F_HW is None:
        F_HW = Rx>=0
    
    dipRad = np.deg2rad(dip)
    
    # Interpolate to get the model coefficients at the required periods
    [c1,c1_a,c1_b,c1_c,c1_d,c_n,c_m,c3,c5,c_HM,c6,c7,c7_b,c8,c8_b,c9,c9_a,
     c9_b,c11_b,c_g1,c_g2,c_g3,phi1,phi2,phi3,phi4,phi5,phi6,gamma_JP_IT,
     gamma_Wn,phi1_JP,phi5_JP,phi6_JP,tau1,tau2,sigma1,sigma2,sigma3,
     sigma2_JP] = __interp_2_vec(T, __Chiou_2014_Periods, __Chiou_2014_Coeff)
    c2 = 1.06
    c4 = -2.1
    c4_a = -0.5
    c_RB = 50
    c8_a = 0.2695
    c11 = 0
    D_DPP = 0
    
    # Region specific adjustments
    if region==2:
        phi1 = phi1_JP
        phi5 = phi5_JP
        phi6 = phi6_JP
        sigma2 = sigma2_JP
    if region in (2, 4) and 6 < M < 6.9:
        gamma_A = gamma_JP_IT
    elif region == 3:
        gamma_A = gamma_Wn
    else:
        gamma_A = 1
    
    # Mean Ztor: Equations (4) and (5)
    if F_RV==1:
        E_Ztor = max(2.704-1.226*max(M-5.849,0),0)**2
    else:
        E_Ztor = max(2.673-1.136*max(M-4.970,0),0)**2
    if Ztor == -999:
        Ztor = E_Ztor
    D_ZTOR = Ztor - E_Ztor
    
    # Median intensity on the reference site (Vs30 = 1130 m/s): Equation (11)
    ln_y_ref = (
            c1 + (c1_a + c1_c/np.cosh(2*max(M-4.5,0))) * F_RV +
            (c1_b + c1_d/np.cosh(2*max(M-4.5,0))) * F_NM + 
            (c7 + c7_b/np.cosh(2*max(M-4.5,0))) * D_ZTOR + 
            (c11 + c11_b/np.cosh(2*max(M-4.5,0)))* np.cos(dipRad)**2 + 
            c2*(M-6) + (c2-c3)/c_n * np.log(1+np.exp(c_n*(c_m-M))) + 
            c4 * np.log(Rrup + c5*np.cosh(c6*np.maximum(M-c_HM,0))) + 
            (c4_a-c4) * np.log(np.sqrt(Rrup**2+c_RB**2)) + 
            gamma_A * (c_g1 + c_g2/(np.cosh(np.maximum(M-c_g3,0)))) * Rrup + 
            c8 * max(1-max(Rrup-40,0)/30,0) * 
            min(max(M-5.5,0)/0.8,1) * np.exp(-c8_a*(M-c8_b)**2) * D_DPP + 
            c9 * F_HW * np.cos(dipRad) * (c9_a+(1-c9_a)*np.tanh(Rx/c9_b)) * 
            (1-np.sqrt(Rjb**2+Ztor**2)/(Rrup+1))
            )   
    y_ref = np.exp(ln_y_ref)
    
    # Equations (1) and (2)
    if region != 2: # California and non-Japan regions
        E_z1 = np.exp(-7.15/4*np.log((Vs30**4+570.94**4)/(1360**4+570.94**4)))
    else: # Japan
        E_z1 = np.exp(-5.23/2*np.log((Vs30**2+412.39**2)/(1360**2+412.39**2)))
    if z1 == -999:
        D_Z1 = 0
    else:
        D_Z1 = z1 - E_z1
    
    # Median intensity: Equation (12)
    ln_y = ( 
            ln_y_ref +
            phi1*min(np.log(Vs30/1130),0) +
            phi2*(np.exp(phi3*(min(Vs30,1130)-360))-np.exp(phi3*(1130-360))) *
            np.log((y_ref+phi4)/phi4) +
            phi5*(1-np.exp(-D_Z1/phi6))
            )
    
    # Standard deviation terms: Equation (13)
    Finferred = 1 - F_Vs30
    Fmeasured = F_Vs30
    NL_0 = (phi2*(np.exp(phi3*(min(Vs30,1130)-360))-np.exp(phi3*(1130-360))) *
            (y_ref/(y_ref+phi4)))
    sigmaNL0 = ((sigma1+(sigma2 - sigma1)/1.5*(min(max(M,5),6.5)-5)) *
                np.sqrt((sigma3*Finferred + 0.7* Fmeasured) + (1+NL_0)**2))
    tau = tau1 + (tau2-tau1)/1.5 * (min(max(M,5),6.5)-5)
    sigma = np.sqrt((1+NL_0)**2*tau**2 + sigmaNL0**2)
    
    inter = tau 
    intra = np.sqrt(sigma**2 - tau**2)
    
    return ln_y, sigma, intra, inter







def __interp_2_vec(z, x, y) :
    col_idx = np.argmax(np.greater.outer(x, z), axis=0) - 1
    ret = (y[col_idx + 1, :] - y[col_idx, :]).T
    ret /= x[col_idx + 1] - x[col_idx]
    ret *= z - x[col_idx]
    ret += y[col_idx, :].T
    return ret


