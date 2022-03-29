# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import numpy as np
import pandas as pd




def readOneFlatfile(fileName):
    '''
    Read a CSV flatfile with the intensities and metadata of ground motion 
    records.
    
    Parameters
    ----------
    fileName : string
        Path of CSV file.
    
    Returns
    ----------
    pga : (n,) numpy array
        Peak ground acceleration of the records.
    pgv : (n,) numpy array
        Peak ground velocity of the records.
    periods : (m,) numpy array
        Periods at which the spectral accelerations are computed.
    Sa : (n,m) numpy array
        Spectral acceleration of the records.
    metadata : dictionary
        Metadata of the records.
    '''
    
    # Read CSV file
    dtype={"Record Sequence Number": int, "EQID": int, 'FW/HW Indicator':str}
    df = pd.read_csv(fileName, dtype=dtype)
    
    # Get metadata
    RSNs = df['Record Sequence Number'].values
    nRec = len(RSNs)
    metadata = dict()
    metadata['magnitude'] = df['Earthquake Magnitude'].values
    metadata['Vs30'] = df['Vs30 (m/s) selected for analysis'].values
    metadata['Rjb'] = df['Joyner-Boore Dist. (km)'].values
    metadata['lowestFreq'] = df['Lowest Usable Freq - Ave. Component (Hz)'].values
    metadata['rake'] = df['Rake Angle (deg)'].values
    metadata['Rrup'] = df['ClstD (km)'].values
    metadata['z1'] = df['Northern CA/Southern CA - H11 Z1 (m)'].values
    metadata['station_lat'] = df['Station Latitude'].values
    metadata['station_lon'] = df['Station Longitude'].values
    metadata['EQID'] = df['EQID'].values
    metadata['dip'] = df['Dip (deg)'].values
    metadata['Rx'] = df['Rx'].values
    metadata['Ztor'] = df['Depth to Top Of Fault Rupture Model'].values
    metadata['fHW'] = df['FW/HW Indicator'].values
    metadata['hyp_lat'] = df['Hypocenter Latitude (deg)'].values
    metadata['hyp_lon'] = df['Hypocenter Longitude (deg)'].values
    
    # Get PGA and PGV
    pga = df['PGA (g)'].values
    pgv = df['PGV (cm/sec)'].values
    
    # Get periods from the column names
    columnNames = np.array(df.columns)
    periodsNames = []
    for name in columnNames:
        if name[0]=='T' and name[-1]=='S':
            periodsNames.append(name)
    nP = len(periodsNames)
    
    # Get spectral accelerations
    Sa = np.zeros((nRec, nP))
    for i in range(nP):
        Sa[:,i] = df[periodsNames[i]].values
    periods = np.array(list(map(float, [i[1:-1] for i in periodsNames])))
    
    return pga, pgv, periods, Sa, metadata



def readMultipleFlatfiles(fileNames):
    '''
    Read a list of CSV flatfile with the intensities and metadata of ground 
    motion records.
    
    Parameters
    ----------
    fileNames : list of strings
        List of paths of CSV files.
    
    Returns
    ----------
    pga : (n,) numpy array
        Peak ground acceleration of the records.
    pgv : (n,) numpy array
        Peak ground velocity of the records.
    periods : (m,) numpy array
        Periods at which the spectral accelerations are computed.
    Sa : (n,m,p) numpy array
        Spectral acceleration of the records.
    metadata : dictionary
        Metadata of the records.
    '''
    
    pga,pgv,periods,firstSa,metadata = readOneFlatfile(fileNames[0])
    
    nXi = len(fileNames)
    Sa = np.zeros(firstSa.shape + (nXi,))
    Sa[:,:,0] = firstSa
    for i in range(1,nXi):
        __,__,__,Sa[:,:,i],__ = readOneFlatfile(fileNames[i])
    
    return pga, pgv, periods, Sa, metadata
    