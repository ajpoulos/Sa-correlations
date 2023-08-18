# -*- coding: utf-8 -*-
"""
@author: Alan Poulos

Example that reproduces part of Figure 8 of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

from correlationModel import correlationModel


# Define variables
nT1 = 100
T1 = np.logspace(np.log10(0.01), np.log10(10), nT1)
T2 = [0.1, 2]
xi = np.array([0.01, 0.05, 0.2])
nXi = len(xi)
nT2 = len(T2)





# Figure 8
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[10, 4.5])
ax1 = axes[0]
ax2 = axes[1]

plotingTs = [0.1, 2]
xi1 = [0.01, 0.05, 0.2]
npXi = len(xi1)

# Same damping ratio for both oscillators
for j in np.arange(npXi-1,-1,-1):
    indj = np.where(xi==xi1[j])[0][0]
    label = r'$\xi_1$ = $\xi_2$ = '+str(int(xi[indj]*100))+r'%'
    for i,pT in enumerate(plotingTs):
        
        # Correlations from regression model
        corrApprox = np.zeros(nT1)
        for k in range(nT1):
            corrApprox[k] = correlationModel(T1[k], xi1[j], pT, xi1[j])
            
        if i==0:
            ax1.semilogx(T1, corrApprox, color='C'+str(j), ls='--', lw=2, label=label)
        else:
            ax1.semilogx(T1, corrApprox, color='C'+str(j), ls='--', lw=2)

ax1.set_xlim(0.01, 10)
ax1.set_ylim(0, 1)
ax1.xaxis.set_major_formatter(formatter)
ax1.set_xlabel('Period, $T_1$ [s]')
ax1.set_ylabel(r'Correlation, $\rho(T_1,\xi_1,T_2,\xi_2)$')
ax1.legend(loc=(0.03,0.47), frameon=False)
ax1.tick_params(right=True, top=True, which='both')
ax1.text(0.01, 1.04, '(a)', transform=ax1.transAxes, fontsize=12)

# Different damping ratio for both oscillators
xi2 = 0.01
indXi2 = np.where(xi==xi2)[0][0]
for j in np.arange(npXi):
    indXi1 = np.where(xi==xi1[j])[0][0]
    label = r'$\xi_1$ = '+str(int(xi1[j]*100))+r'%, $\xi_2$ = 1%'
    for i,pT in enumerate(plotingTs):
        
        # Correlations from regression model
        corrApprox = np.zeros(nT1)
        for k in range(nT1):
            corrApprox[k] = correlationModel(T1[k], xi1[j], pT, xi2)
        
        if i==0:
            ax2.semilogx(T1, corrApprox, color='C'+str(j), ls='--', lw=2, label=label)
        else:
            ax2.semilogx(T1, corrApprox, color='C'+str(j), ls='--', lw=2)
        
ax2.set_xlabel('Period, $T_1$ [s]')
ax2.xaxis.set_major_formatter(formatter)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], frameon=False, loc=(0.03,0.42))
ax2.tick_params(right=True, top=True, which='both')
ax2.text(0.01, 1.04, '(b)', transform=ax2.transAxes, fontsize=12)

plt.tight_layout()