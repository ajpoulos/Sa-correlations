# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

from correlationModel import correlationModel



# Load computed correlations
results = pickle.load(open("correlationResults.pkl", "rb"))
correlations = results['correlations']
correlations_PGV = results['correlations_PGV']
xi = results['xi']
T = results['T']


# Precomputations
nXi = len(xi)
nT = len(T)
plotingTs = [0.1, 2]
Tmin, Tmax = T[0], T[-1]
ind_5p = np.where(xi==0.05)[0][0]
T1,T2 = np.meshgrid(T, T)
Z = correlations[ind_5p,ind_5p]
Xi1,Xi2 = np.meshgrid(xi, xi)


# Settings
condT = 1
ind_condT = np.where(T==condT)[0][0]
colors = ['C0', 'C1', 'C2', 'C3', 'k', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']




# # Figure 5
# fig = plt.figure()
# ax = plt.gca()
# handles = []
# rho_5 = correlations[ind_5p,ind_5p,ind_condT,:]
# for j in np.arange(nXi-1,-1,-1):
#     indXi = np.where(xi==xi[j])[0][0]
#     corr_x = correlations[indXi,indXi,ind_condT,:]
#     h, = plt.semilogx(T, corr_x - rho_5, color=colors[j], ls='-')
#     handles.append(h)

# plt.xlim(Tmin, Tmax)
# plt.ylim(-0.3, 0.3)
# plt.gca().xaxis.set_major_formatter(formatter)
# plt.gca().tick_params(right=True, top=True, which='both')
# plt.xlabel('Period, $T_1$ [s]')
# text = r'$\rho(T_1,\xi,1\mathrm{s},\xi)$ - $\rho_{5\%}(T_1,1\mathrm{s})$'
# plt.text(-0.07, 1.04, text, ha='left', va='bottom', transform=ax.transAxes)

# labels = [r'$\xi$ = ' + str(int(xi[i]*100)) + '%' for i in range(nXi)]
# labels[0] = r'$\xi$ = 0.5%'
# legend = fig.legend(handles, labels[::-1], loc='center left', 
#                     bbox_to_anchor=(0.98, 0.5), fancybox=False, edgecolor='k', 
#                     title='Damping ratios')
# legend.get_frame().set_linewidth(0.8)
# plt.tight_layout()




# # Figure 6
# size = np.array([7.4, 4.5])
# T1 = 1
# T2s = [0.1, 0.3, 1, 3]
# vmins = [0,     0.45,   0.9,    0.55]
# vmaxs = [0.5,   0.75,   1.0,    0.80]
# letters = ['(a)', '(b)', '(c)', '(d)']

# fig = plt.figure(figsize=2*size)
# indT1 = np.where(T==T1)[0][0]

# npT = len(T2s)
# for i in range(npT):
    
#     indT2 = np.where(T==T2s[i])[0][0]
#     Z = correlations[:,:,indT1,indT2]
    
#     ax = fig.add_subplot(2, 2, i+1, projection='3d')
#     ax.view_init(azim=210)
#     surf = ax.plot_surface(np.log10(Xi1*100), np.log10(Xi2*100), Z, 
#                            cmap=coolwarm, linewidth=0.2, ec='k', vmin=vmins[i], 
#                            vmax=vmaxs[i])
#     plt.xlabel(r'$\xi$ for T = '+str(T2s[i])+' s [%]')
#     plt.ylabel(r'$\xi$ for T = '+str(T1)+' s [%]')
#     ticksL = np.array([1, 3, 10, 30])
#     ticks = np.log10(ticksL)
#     ax.set_yticks(ticks)
#     ax.set_yticklabels(ticksL)
#     ax.set_xticks(ticks)
#     ax.set_xticklabels(ticksL)
#     ax.set_xlim(np.log10(0.5), np.log10(30))
#     ax.set_ylim(np.log10(0.5), np.log10(30))
#     ax.set_zlim(vmins[i], vmaxs[i])
#     fig.colorbar(surf, ax=ax, label='Correlation', shrink=0.85)
#     ax.text2D(0.05, 0.9, letters[i], transform=ax.transAxes, fontsize=12)
# plt.subplots_adjust(wspace=-0.2)




# # Figure 7
# fig = plt.figure(figsize=(7,6))
# ax = fig.add_subplot(projection='3d')
# ax.view_init(azim=245)

# X,Y = np.meshgrid(T, xi)

# surf = ax.plot_surface(np.log10(X), np.log10(Y*100), correlations_PGV, ec='k', 
#                        cmap=coolwarm, linewidth=0.3, vmin=0.3, vmax=0.9)
# ax.plot(np.log10(T), np.log10(5)*np.ones(nT), correlations_PGV[ind_5p], 
#         color='k', zorder=3, lw=1.5)
# fig.colorbar(surf, label='Correlation', shrink=0.7)
# plt.xlabel('Period [s]')
# plt.ylabel('Damping ratio [%]')
# ticksL_y = np.array([1, 3, 10, 30])
# ticksL_x = np.array([0.01, 0.1, 1, 10])
# ax.set_yticks(np.log10(ticksL_y))
# ax.set_yticklabels(ticksL_y)
# ax.set_xticks(np.log10(ticksL_x))
# ax.set_xticklabels(ticksL_x)
# ax.set_xlim(np.log10(0.01), np.log10(10))
# ax.set_ylim(np.log10(0.5), np.log10(30))
# ax.set_zlim(0, 1)




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
    for i,pT in enumerate(plotingTs):
        ind = np.where(T==pT)[0][0]
        
        # Correlations from data
        corr = correlations[indj,indj,ind,:]
        if i==0:
            label = r'$\xi_1$ = $\xi_2$ = '+str(int(xi[indj]*100))+r'%'
            ax1.semilogx(T, corr, label=label, color='C'+str(j), ls='-')
        else:
            ax1.semilogx(T, corr, color='C'+str(j), ls='-')
        
        # Correlations from regression model
        corrApprox = np.zeros(nT)
        for k in range(nT):
            corrApprox[k] = correlationModel(T[k], xi1[j], pT, xi1[j])
        ax1.semilogx(T, corrApprox, color='C'+str(j), ls='--', lw=2)

ax1.set_xlim(Tmin, Tmax)
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
    for i,pT in enumerate(plotingTs):
        indT2 = np.where(T==pT)[0][0]
        
        # Correlations from data
        corr = correlations[indXi1,indXi2,:,indT2]
        if i==0:
            label = r'$\xi_1$ = '+str(int(xi1[j]*100))+r'%, $\xi_2$ = 1%'
            ax2.semilogx(T, corr, label=label, color='C'+str(j), ls='-')
        else:
            ax2.semilogx(T, corr, color='C'+str(j), ls='-')
        
        # Correlations from regression model
        corrApprox = np.zeros(nT)
        for k in range(nT):
            corrApprox[k] = correlationModel(T[k], xi1[j], pT, xi2)
        ax2.semilogx(T, corrApprox, color='C'+str(j), ls='--', lw=2)
        
ax2.set_xlabel('Period, $T_1$ [s]')
ax2.xaxis.set_major_formatter(formatter)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], frameon=False, loc=(0.03,0.42))
ax2.tick_params(right=True, top=True, which='both')
ax2.text(0.01, 1.04, '(b)', transform=ax2.transAxes, fontsize=12)

plt.tight_layout()