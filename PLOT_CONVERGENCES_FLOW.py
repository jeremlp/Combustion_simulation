# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:02:37 2024

@author: jerem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

plt.close('all')

NAME_FILE = 'FIELD_PLOT/FLOW_FINAL_C0.05_CONV-2.txt'
NAME_SAVE = NAME_FILE[:-3] + "png"

pf = pd.read_csv(NAME_FILE)
arr = pf.to_numpy().T

def fit(x, a,b,c):
    return a-c*x**(-b)
def fit_inv(x, a,b,c):
    return c*x**(-b) + a
L = 0.002 #2 mm

for k,l in enumerate(arr[1:]):
    l = l.astype(float)
    Nx = arr[0].astype(float)
    err = L/(Nx-1)*1000
    mask = ~np.isnan(l)

    plt.plot(arr[0][mask],l[mask],"o-", label = pf.columns[k+1])
    print(k)
    if k !=3 and k!=4:
        j = 2
        try:
            popt, pcov = curve_fit(fit,Nx[mask][j:], l[mask][j:], [3100,1,3100])
        except:
            popt, pcov = curve_fit(fit_inv,Nx[mask][j:], l[mask][j:], [1500,2,3000])
        # popt, pcov = curve_fit(func, Nx[mask][j:], l[mask][j:],[1000,0.1,1490])
        plt.plot(Nx[j:],fit(Nx[j:], *popt),"k--",label =f"{popt[0]:.1f} +-{ np.sqrt(pcov[0,0]):.1f}")

plt.grid()
plt.xlabel("Nx")
# plt.ylabel("Length of diffusion zone of N2 (10% - 90%) [mm]")
plt.ylabel("Strain rate on left wall [s-1]")
plt.legend()
plt.title("Grid Convergence of velocity advection (C=0.05)")
plt.tight_layout()
# plt.savefig(NAME_SAVE)


#======================================
# ORDER OF METHOD
#======================================



plt.figure()
plt.grid()
plt.xscale("log")
plt.yscale("log")

for name in ["Upwind", "Lax_Wendroff","LUD_LW"]:
    if name=="LUD_LW":
        y = pf[name][:-2].astype(float) - 1486
        x = Nx[:-2]
    else:
        y = pf[name]-1486
        x = Nx

    a,b = np.polyfit(np.log(x[4:]),np.log(y[4:]),1)
    popt, pcov = curve_fit(fit,x, y, [3100,0.1,0])
    plt.plot(x,y,".-",label = f"{name} : {a:.2f}")
    plt.plot(x[4:],np.exp(a*np.log(x[4:])+b),"b--")
    # plt.plot(x,func(x, *popt),"r--",label =f"{popt[-1]:.1f} +-{ np.sqrt(pcov[-1,-1]):.1f}")
    plt.legend()

plt.figure()
plt.xlabel("Nx")
plt.ylabel(r"$\epsilon-\epsilon^*$", fontsize=12)
plt.title("Order of methods for the maximum gradient $|\partial Y_{N2}/\partial y|$")
plt.xscale("log")
plt.yscale("log")

for name in pf.columns[1:]: #:
    if name=="LUD_LW":
        eps = pf[name][:-2].to_numpy().astype(float)
        x = Nx[:-2]
    elif name=="RK3_LW":
        eps = pf[name][:-3].to_numpy().astype(float)
        x = Nx[:-3]
    elif name== "RK3_LUD_Upwind":
        continue
    else:
        eps = pf[name].to_numpy()
        x = Nx
    try:
        popt, pcov = curve_fit(fit,x[0:], eps[0:], [3100,1,3100])
    except:
        popt, pcov = curve_fit(fit_inv,x[4:], eps[4:], [1500,2,3000])

    estim = popt[0]
    y = np.abs(popt[0]-eps)
    # a,b = np.polyfit(np.log(x[start:]),np.log(y[start:]),1)
    plt.plot(x,y,".-", label = f"{name} : ${popt[1]:.2f} \pm{ np.sqrt(pcov[1,1]):.2f}$")

    # plt.plot(x,y,".-",label = f"{name} : {a:.2f}")
    # plt.plot(x[start:],np.exp(a*np.log(x[start:])+b),"k--")
    plt.plot(x,np.abs(estim-fit(x, *popt)),"k--")
    print(name,popt[0], popt[1])
    # print(vie(Nx,eps))
plt.legend()

