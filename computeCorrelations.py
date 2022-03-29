# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import os
import pickle
import numpy as np

from Codes.readFlatfiles import readMultipleFlatfiles
from Codes.Chiou2014 import Chiou_2014
from Codes.dampingModificationModel import Rezaeian2014
from Codes.helper_func import faultType, get_CY_Region, mixedEffects
from Codes.correlation import nanCorrcoef3D, nanCorrcoef3D_1D


# Settings
xi = np.array([0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30])
nXi = len(xi)
Rmax = 100
F_Vs30 = 1


# Load NGA-West2 flatfiles
base = os.path.join('Flatfiles', 'NGA_West2_d')
flatfiles = [base+str(int(thisXi*1000)).zfill(3)+'.csv' for thisXi in xi]
pga, pgv, T, Sa, metadata = readMultipleFlatfiles(flatfiles)
M = metadata['magnitude']
Vs30 = metadata['Vs30']
Rjb = metadata['Rjb']
Rrup = metadata['Rrup']
Rx = metadata['Rx']
Ztor = metadata['Ztor']
lowestFreq = metadata['lowestFreq']
rake = metadata['rake']
z1 = metadata['z1']
hyp_lat = metadata['hyp_lat']
hyp_lon = metadata['hyp_lon']
EQID = metadata['EQID']
dip = metadata['dip']
fHW = metadata['fHW']
F_HW = 1*(fHW=='hw')
nT = len(T)
nUsed = len(pga)


# Obtain the region of each record
regions = get_CY_Region(hyp_lon, hyp_lat)


# Compute the total residual and logarithmic standard deviations of each record
totalRes = np.zeros((nXi, nUsed, nT))
sigma = np.zeros((nXi, nUsed, nT))
phi = np.zeros((nXi, nUsed, nT))
tau = np.zeros((nXi, nUsed, nT))
totalRes_PGV = np.zeros(nUsed)
sigma_PGV = np.zeros(nUsed)
phi_PGV = np.zeros(nUsed)
tau_PGV = np.zeros(nUsed)
for i in range(nUsed):
    
    F = faultType(rake[i])
    
    #PGV
    lnPGV,sigmaPGV,phiPGV,tauPGV = Chiou_2014(M[i], Rjb[i], Rrup[i], Rx[i], 
                        Ztor[i], np.array([-1]), Vs30[i], F, dip[i], z1[i], 
                        regions[i], F_Vs30, F_HW[i])
    totalRes_PGV[i] = np.log(pgv[i]) - lnPGV[0]
    sigma_PGV[i] = sigmaPGV[0]
    phi_PGV[i] = phiPGV[0]
    tau_PGV[i] = tauPGV[0]
    
    # 5% damping case
    lnSa_5,sigma_5,phi_5,tau_5 = Chiou_2014(M[i], Rjb[i], Rrup[i], Rx[i], 
                       Ztor[i], T, Vs30[i], F, dip[i], z1[i], regions[i],
                       F_Vs30, F_HW[i])
    
    # All damping ratios
    for k in range(nXi):
        
        # Get damping scaling factors
        lnDSF,sigma_lnDSF,rho_lnDSF = Rezaeian2014(xi[k]*100, M[i], Rrup[i], T)
        if xi[k]==0.05:
            rho_lnDSF = np.zeros(nT)
            lnDSF = np.zeros(nT)
            
        # Modify logarithmic mean
        lnSa = lnSa_5 + lnDSF
        
        # Modify logarithmic standard deviation
        sigma_xi = np.sqrt(sigma_5**2 + sigma_lnDSF**2 + 
                           2*sigma_5*sigma_lnDSF*rho_lnDSF)
        sigma[k,i,:] = sigma_xi
        phi[k,i,:] = phi_5 * sigma_xi/sigma_5
        tau[k,i,:] = tau_5 * sigma_xi/sigma_5
        
        # Total residual
        totalRes[k,i,:] = np.log(Sa[i,:,k]) - lnSa

totalResNorm = totalRes / sigma
totalResNorm_PGV = totalRes_PGV / sigma_PGV





# Compute intra and inter event residuals
events, EQ_indices = np.unique(EQID, return_index=True)
nEvents = len(events)
intraRes = np.zeros((nXi, nUsed, nT))
interRes = np.nan*np.ones((nXi, nEvents, nT))
interResNorm = np.nan*np.ones((nXi, nEvents, nT))
intraRes_PGV = np.zeros(nUsed)
interRes_PGV = np.nan*np.zeros(nEvents)
interResNorm_PGV = np.nan*np.zeros(nEvents)

for i in range(nEvents):
    
    # Find records of the given event
    where = np.where(EQID==events[i])[0]
    
    # Compute PGV residuals
    thisTau_PGV = tau_PGV[where[-1]]
    eta_PGV, this_intraRes_PGV = mixedEffects(totalRes_PGV[where], thisTau_PGV, 
                                              phi_PGV[where])
    intraRes_PGV[where] = totalRes_PGV[where] - eta_PGV
    if len(where)>1:
        interRes_PGV[i] = eta_PGV
        interResNorm_PGV[i] = eta_PGV/thisTau_PGV
    
    # Compute Sa residuals
    for k in range(nT):
        
        filt_ind = np.where(1/T[k] >= lowestFreq[where])[0]
        if len(filt_ind)>0:
            where_filt = where[filt_ind]
            
            for j in range(nXi):
                thisTau = tau[j,where_filt[-1],k]
                eta, this_intraRes = mixedEffects(totalRes[j,where_filt,k], 
                                                  thisTau, phi[j,where_filt,k])
                intraRes[j,where,k] = totalRes[j,where,k] - eta
                if len(filt_ind)>1:
                    interRes[j,i,k] = eta
                    interResNorm[j,i,k] = eta/thisTau

intraResNorm = intraRes/phi
intraResNorm_PGV = intraRes_PGV/phi_PGV



    
# Set residuals to np.nan where period is greater than maximum usable
for i in range(nUsed):
    if lowestFreq[i]>0:
        nan_ind = np.where(1/T < lowestFreq[i])[0]
        totalResNorm[:,i,nan_ind] = np.nan
        intraResNorm[:,i,nan_ind] = np.nan

# Set residual to np.nan where distance is greater than the maximum distance
totalResNorm[:,Rjb>Rmax,:] = np.nan
intraResNorm[:,Rjb>Rmax,:] = np.nan
totalResNorm_PGV[Rjb>Rmax] = np.nan
intraResNorm_PGV[Rjb>Rmax] = np.nan



# Compute correlations
correlations_intra = nanCorrcoef3D(intraResNorm)
correlations_inter = nanCorrcoef3D(interResNorm)
correlations_intra_PGV = nanCorrcoef3D_1D(intraResNorm, intraResNorm_PGV)
correlations_inter_PGV = nanCorrcoef3D_1D(interResNorm, interResNorm_PGV)


# Number of usable residuals for each period combination
nUsable_PGV = np.sum(~np.isnan(intraResNorm[0,:,:]), axis=0)
nUsable = np.minimum.outer(nUsable_PGV, nUsable_PGV)



# Use standard deviations of a "typical" event
M_t = 7; R_t = 20; Vs30_t = 500; F_t = 3; z1_t = 1000; region_t = 1; Ztor_t = 1
dip_t = 90 ;F_Vs30_t = 1 ;F_HW_t = 0

__,__,phi_Sa_typ,tau_Sa_typ = Chiou_2014(M_t, R_t, R_t, R_t, Ztor_t, T, Vs30_t, 
                                         F_t, dip_t, z1_t, region_t, F_Vs30_t, 
                                         F_HW_t)
__,__,phi_PGV_t,tau_PGV_t = Chiou_2014(M_t, R_t, R_t, R_t, Ztor_t,
                                       np.array([-1]), Vs30_t, F_t, dip_t, 
                                       z1_t, region_t, F_Vs30_t, F_HW_t)

phi_PGV_typ = phi_PGV_t[0]
tau_PGV_typ = tau_PGV_t[0]
sigma_Sa_typ = np.sqrt(phi_Sa_typ**2 + tau_Sa_typ**2)
sigma_PGV_typ = np.sqrt(phi_PGV_typ**2 + tau_PGV_typ**2)


# Combine inter and intra event correlations
correlations = ((np.outer(tau_Sa_typ, tau_Sa_typ) * correlations_inter + 
                  np.outer(phi_Sa_typ, phi_Sa_typ) * correlations_intra) / 
                np.outer(sigma_Sa_typ, sigma_Sa_typ))
correlations_PGV = ((tau_Sa_typ*tau_PGV_typ*correlations_inter_PGV +
                      phi_Sa_typ*phi_PGV_typ*correlations_intra_PGV) / 
                    (sigma_Sa_typ*sigma_PGV_typ))




# Save results
results = {'correlations':correlations, 'correlations_PGV':correlations_PGV, 
          'nUsable':nUsable, 'nUsable_PGV':nUsable_PGV, 'xi':xi, 'T':T}
pickle.dump(results, open("correlationResults.pkl", "wb"))
