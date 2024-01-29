# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 14:00:42 2024

@author: jerem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.close('all')

NAME_FILE = 'MASS_PLOT/DERIV_C0.1_TRUE_AIR.txt'
NAME_SAVE = NAME_FILE[:-3] + "png"

pf = pd.read_csv(NAME_FILE)
arr = pf.to_numpy().T


L = 0.002 #2 mm

for k,l in enumerate(arr[1:]):
    name = pf.columns[k+1]
    if name == "RK3_WENO" or name=="RK3_LUD": continue

    Nx = arr[0]
    err = L/(Nx-1)*1000
    plt.plot(arr[0],l,"o-", label = pf.columns[k+1])

plt.grid()
plt.xlabel("Nx")
# plt.ylabel("Length of diffusion zone of N2 (10% - 90%) [mm]")
plt.ylabel("Slope of N2 species diffusion \n on left wall [m-1]")
plt.legend()
plt.title("Grid Convergence of mass advection (C=0.1)")
plt.tight_layout()
plt.savefig(NAME_SAVE)

plt.figure()
L = 0.002 #2 mm

# NAME_FILE = 'MASS_PLOT/LENGTH_C0.1_TRUE_AIR.txt'
# NAME_SAVE = NAME_FILE[:-3] + "png"

# pf = pd.read_csv(NAME_FILE)
# arr = pf.to_numpy().T

# for k,l in enumerate(arr[1:]):
#     if k == 6: continue
#     Nx = arr[0]
#     err = 2*L/(Nx-1)*1000
#     plt.plot(arr[0],l,"o-", label = pf.columns[k+1])
#     plt.errorbar(arr[0],l, err,linestyle='None', fmt='None', capsize=5, color='black',alpha=1)


NAME_FILE = 'MASS_PLOT/LENGTH_C0.1_INTERPOLATION.txt'
NAME_SAVE = NAME_FILE[:-3] + "png"

pf = pd.read_csv(NAME_FILE)
arr = pf.to_numpy().T
for k,l in enumerate(arr[1:]):
    name = pf.columns[k+1]
    if name == "RK3_WENO" or name=="RK3_LUD": continue

    Nx = arr[0]
    err = L/(Nx-1)*1000
    plt.plot(arr[0],l,"o-", label = pf.columns[k+1])


NAME_FILE = 'MASS_PLOT/LENGTH_C0.1_INTERPOLATION_2.txt'
NAME_SAVE = NAME_FILE[:-3] + "png"

pf = pd.read_csv(NAME_FILE)
pf=pf.reindex(columns=["Nx",   "RK3_LUD",    "WENO", "RK3_LW"      ])
arr = pf.to_numpy().T
for k,l in enumerate(arr[1:]):
    name = pf.columns[k+1]
    if name == "RK3_WENO" or name=="RK3_LUD": continue

    Nx = arr[0]
    err = L/(Nx-1)*1000
    plt.plot(arr[0],l,"o-", label = pf.columns[k+1])

NAME_FILE = 'MASS_PLOT/LENGTH_C0.1_INTERPOLATION_3.txt'
NAME_SAVE = NAME_FILE[:-3] + "png"

pf = pd.read_csv(NAME_FILE)
arr = pf.to_numpy().T
for k,l in enumerate(arr[1:]):
    name = pf.columns[k+1]
    if name == "RK3_WENO" or name=="RK3_LUD": continue

    Nx = arr[0]
    err = L/(Nx-1)*1000
    plt.plot(arr[0],l,"o-", label = name)
# plt.plot(arr[0][1::3],arr[1][1::3],"k--")
# plt.plot(arr[0][::3],l[::3],"r--",)
# plt.plot(arr[0][1::3],arr[-1][1::3],"k--")
# plt.plot(arr[0],np.mean(np.vstack([arr[1],arr[-1]]),0), "r")


# plt.errorbar(arr[0][::3],l[::3], err[::3],linestyle='None', fmt='None', capsize=5, color='black',alpha=1)
plt.ylim(0.245,0.325)
plt.grid()
plt.xlabel("Nx")
plt.ylabel("Length of diffusion zone of N2 (10% - 90%) [mm]")
# plt.ylabel("Slope of N2 species diffusion \n on left wall")
plt.legend()
plt.title("Grid Convergence of mass advection (C=0.1)")
plt.tight_layout()

plt.savefig(NAME_SAVE)

# plt.figure()
# plt.plot(arr[0][1::3],arr[1][1::3],"k-.", label = "Upwind/LUD")
# plt.plot(arr[0][::3],l[::3],"r--",label = "Match of all methods")
# plt.plot(arr[0][1::3],arr[-3][1::3],"k--", label = "LW/WENO")
# plt.plot(arr[0],np.mean(np.vstack([arr[1],arr[-3]]),0), "r", label = "Mean Upwind - WENO")

# plt.errorbar(arr[0][::3],arr[-1][::3], err[::3],linestyle='None', fmt='None', capsize=5, color='black',alpha=1)


# plt.grid()
# plt.xlabel("Nx")
# plt.ylabel("Length of diffusion zone of N2 (10% - 90%) [mm]")
# # plt.ylabel("Slope of N2 species diffusion \n on left wall")
# plt.legend()
# plt.title("Grid Convergence of mass advection (C=0.1)")
# plt.tight_layout()

# plt.savefig("MASS_PLOT/LENGTH_C0.1_TRUE_AIR_MEAN.png")


plt.figure()
NAME_FILE = 'MASS_PLOT/DERIV_C0.1_TRUE_AIR.txt'
NAME_SAVE = NAME_FILE[:-3] + "png"


#======================================
# ORDER OF METHOD
#======================================
from scipy.optimize import curve_fit


def vie(k,eps):
    h = 1/(k-1)
    p = -1/np.log(2) * np.log((eps[-1] - eps[-3])/(eps[-3]-eps[-5]))
    C = (eps[-1] - eps[-3])/(h[-1] - h[-3])
    estim = (eps[-1] - 0.5**p*eps[-3])/(1-0.5**p)

    return p, C, estim

def func(x, a, b, c):
    return  c-a * np.exp(-b * x)

def fit(x, a,b,c):
    return a-c*x**-b
pf_deriv = pd.read_csv(NAME_FILE)
plt.grid()
plt.xscale("log")
plt.yscale("log")

start = 4
for name in pf_deriv.columns[1:]: #:
    if name == "RK3_WENO" or name == "RK3_LUD": continue
    eps = pf_deriv[name].to_numpy()
    popt, pcov = curve_fit(fit,Nx, pf_deriv[name], [3100,1,629547])

    estim = popt[0]
    y = popt[0]-eps
    x = Nx
    # a,b = np.polyfit(np.log(x[start:]),np.log(y[start:]),1)
    plt.plot(x,y,".-", label = f"{name} : ${popt[1]:.2f} \pm{ np.sqrt(pcov[1,1]):.2f}$")

    # plt.plot(x,y,".-",label = f"{name} : {a:.2f}")
    # plt.plot(x[start:],np.exp(a*np.log(x[start:])+b),"k--")
    plt.plot(x,estim-fit(x, *popt),"k--")
    print(name,popt[0])
    # print(vie(Nx,eps))
plt.legend()
plt.xlabel("Nx")
plt.ylabel(r"$\epsilon-\epsilon^*$", fontsize=12)
plt.title("Order of methods for the maximum gradient $|\partial Y_{N2}/\partial y|$")