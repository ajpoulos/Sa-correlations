# -*- coding: utf-8 -*-
"""
@author: Alan Poulos

Example that reproduces part of Figure 8a of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from correlationModel import correlationModel


# Define variables
nT1 = 100
T1 = np.logspace(np.log10(0.01), np.log10(10), nT1)
T2 = [0.1, 2]
xi = np.array([0.01, 0.05, 0.2])
nXi = len(xi)
nT2 = len(T2)



# Compute correlations using the model
rho = np.zeros((nXi, nT2, nT1))
for k in range(nXi):
    for i in range(nT2):
        for j in range(nT1):
            rho[k,i,j] = correlationModel(T1[j], xi[k], T2[i], xi[k])


# Plot correlations
plt.figure()
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
for k in range(nXi-1,-1,-1):
    for i in range(nT2):
        if i==0:
            label = r'$\xi_1$ = $\xi_2$ = '+str(int(xi[k]*100))+r'%'
            plt.semilogx(T1, rho[k,i], color='C'+str(k), ls='--', label=label)
        else:
            plt.semilogx(T1, rho[k,i], color='C'+str(k), ls='--')
plt.xlim(0.01, 10)
plt.ylim(0, 1)
plt.legend(frameon=False, loc=(0.03,0.47))
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlabel(r'Period, $T_1$ [s]')
plt.ylabel(r'Correlation')