# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:14:29 2023

@author: jerem
"""

import math
import numpy as np
from numba import njit
import time
from numpy import sin, cos, pi, exp, sqrt

from tqdm import tqdm

from P3_MODULES import Centered, LUD, Lax_Wendroff, Upwind, iteration_SOR,iteration_GS
from P3_MODULES import deriv_x, deriv_y, laplace, div
from P3_MODULES import import_field, CFL_dt
from WENO_PROJECT3 import RK3_mass_WENO, RK3_mass_LW, RK3_WENO_LW,RK3_LW_ONLY,WENO, RK3_LUD_Upwind,RK3_LUD_LW_FULL,RK3_WENO_LW_FULL
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
plt.close("all")


def solve_mass_Upwind(F):
    for f in F:
        f[1:-1,1:-1] +=  Upwind(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
    return F

def solve_mass_LUD(F):
    for f in F:
        f[1:-1,1:-1] +=  Upwind(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
        f[2:-2,2:-2] += LUD(f, ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt) + D*laplace(f[1:-1,1:-1],dx)*dt
    return F

def solve_mass_LW(F):
    for f in F:
        f[1:-1,1:-1] +=  Lax_Wendroff(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
    return F

def solve_mass_RK3_WENO(F):
    for f in F:
        f[1:-1,1:-1] = RK3_WENO_LW(f, ux, uy, dx,dt,D)
    return F

def solve_mass_RK3_LW(F):
    for f in F:
        f[1:-1,1:-1] = RK3_LW_ONLY(f, ux, uy, dx,dt,D)
    return F

def solve_mass_RK3_LUD(F):
    for f in F:
        f[1:-1,1:-1] = RK3_LUD_Upwind(f, ux, uy, dx,dt,D)
    return F
def solve_mass_WENO(F):
    for f in F:
        f[1:-1,1:-1] += Lax_Wendroff(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
        f[3:-3,3:-3] += WENO(f, ux, uy, dx,dt) + D*laplace(f[2:-2,2:-2],dx)*dt
    return F
def solve_mass_RK3_LUD_FULL(F):
    for f in F:
        f[1:-1,1:-1] = RK3_LUD_LW_FULL(f, ux, uy, dx,dt,D)
    return F

def solve_mass_RK3_WENO_FULL(F):
    for f in F:
        f[1:-1,1:-1] = RK3_WENO_LW_FULL(f, ux, uy, dx,dt,D)
    return F


def inject_mass(F):
    #f_N2, f_O2, f_CH4, f_H2O, f_CO2
    for i, f in enumerate(F):
        f[ 0, 1:Nx//4] =     [0.78, 0.22, 0, 0, 0][i] #AIR MIXTURE
        f[ 0, Nx//4:Nx//2] = [1,    0,    0, 0 ,0][i] #NITROGEN

        f[-1, 1:Nx//4] =     [0,    0,    1, 0, 0][i] #METHANE
        f[-1, Nx//4:Nx//2] = [1,    0,    0, 0, 0][i] #NITROGEN
    return F

def SIMU_MASS(METHOD):
    F = np.zeros((Nfluid, Nx, Nx))
    F[0,:,:] = 0.78
    F[1,:,:] = 0.22 #FILL WITH AIR
    f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F
    EXIT_MASS_LIST = []
    MEAN_MASS_LIST = []
    MID_DERIV_LIST = []
    DIFFU_LENGTH_LIST = []

    COND_MEAN_MASS = []
    COND_EXIT_MASS = []
    COND_DIFFU_LENGTH = []
    convergence_count = 0
    for k,t in enumerate(t_range):
        # print(round(t,5))
        t0 = time.perf_counter()
        F = inject_mass(F)
        F = METHOD(F)
        F[:, :,0]  = F[:, :,1]
        F[:, :,-1]  = F[:, :,-2]
        F[:, 0,:]  = F[:, 1,:]
        F[:, -1,:] = F[:, -2,:]
        f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F

        #INTERPOLATION:
        y = f_N2[:,0]
        x_mm = x_range
        order = y.argsort()
        y_data = y[order]
        x_data = x_mm[order]
        diffu_length = np.interp(0.1*np.max(y), y_data, x_data,  left=None, right=None, period=None)-np.interp(0.9*np.max(y), y_data, x_data,  left=None, right=None, period=None)

        DIFFU_LENGTH_LIST.append(diffu_length*1000)
        # diffu_zone = np.where((f_N2[:,0] < 0.9*np.max(f_N2[:,0])) & (f_N2[:,0] > 0.1*np.max(f_N2[:,0])))[0]
        # if len(diffu_zone) > 1:
        #     diffu_length = np.abs(x_range[diffu_zone[-1]] - x_range[diffu_zone[0]])
        #     # print(round(diffu_length*1000,4),"mm")
        #     DIFFU_LENGTH_LIST.append(diffu_length*1000)
        #     # if t > 3*10**-3 and DIFFU_LENGTH_LIST[-1] == DIFFU_LENGTH_LIST[-2]:
        #     #     convergence_count +=1
        #     #     # print(t*1000,"ms |",conv_count, "|",MID_DERIV_LIST[-1])
        #     #     if convergence_count > 500:
        #     #         return F, diffu_length*1000, k, DIFFU_LENGTH_LIST, f_N2[:,0]
        #     # else:
        #     #     conv_count = 0

    return F, diffu_length*1000, k, DIFFU_LENGTH_LIST, f_N2[:,0]

#==================
# FIELD PARAMETERS
#==================
method_name = "Lax_Wendroff"
Nx = 150
C = 0.05
CONV_COND = 10**-0

#==================
# CONSTANTS
#==================
rdxo0 = 1.1614
D = 15*10**-6
nu = D
L =  0.002 #2*10**-3

Tmax = 8*10**-3

tau_diffu = L**2/D

x0, y0 = L/2,L/2

Nfluid = 5
#===============================================================================================
x_range = np.linspace(0, L, Nx)
dx = x_range[1] - x_range[0]
dt = np.min([CFL_dt(dx, D, C=C),0.95*dx])
t_range = np.arange(0,Tmax, dt)

XX,YY = np.meshgrid(x_range,x_range)


# plt.quiver(XX, YY, ux, uy, norm)
# plt.streamplot(XX, YY, ux, uy, color = norm,density = 0.25, broken_streamlines = False)



f0 = np.zeros((Nx, Nx))

#==========================================================

Nx_range = np.arange(30,140,10)


fig, (ax1,ax2) = plt.subplots(2)
ax1.set_xlabel('Nx')
ax1.set_ylabel('Convergence of\nquantities')

ax1.grid()
ax1.set_xlim(Nx_range[0]*0.9,Nx_range[-1]*1.1)
ax1.set_ylim(0.225,0.35)
# ax1.set_ylim(10**-7,10**7)
# ax1.set_yscale("log")


ax2.set_xlabel('t (ms)')
ax2.set_ylabel('Average of conditions \n (Strain, exit vel, mean norm)')

ax2.grid()
ax2.set_xlim(0,10)
# ax2.set_ylim(10**-8,10**6)
ax2.set_ylim(0.225,0.35)

# ax2.set_yscale("log")
ax2.axhline(CONV_COND, color = "k", ls ="--")
fig.tight_layout()
plt.pause(0.01)


SAVE_ARR_LENGTH = Nx_range
SAVE_ARR_DERIV = Nx_range
headers = ["Nx"]

METHOD_LIST = [solve_mass_Upwind, solve_mass_LUD, solve_mass_LW,solve_mass_LW,
               solve_mass_RK3_WENO, solve_mass_RK3_LW, solve_mass_RK3_LUD,
               solve_mass_WENO, solve_mass_RK3_LUD_FULL, solve_mass_RK3_WENO_FULL]


for METHOD in [solve_mass_LW,solve_mass_Upwind]:
    for CONV_COND_MASS in [10**-2]:
        for C in [0.2]:
            method_name = str(METHOD).split(" ")[1][11:]
            headers.append(method_name)

            line_diffu_length, = ax1.plot([],[], ".-",label = f"{method_name} ({C})")
            # line_exit_mass, = ax1.plot([],[], "-",label = f"exit vel ({method_name})")
            # line_mean_mass, = ax1.plot([],[],"-", label = f"mean norm ({method_name})")

            DIFFU_LENGTH = []
            DIFFU_DERIV = []

            print("=============",method_name,"==============")
            for j,Nx in enumerate(tqdm(Nx_range)):
                print(Nx)
                # print("====================")
                # print(f"NX = {Nx} ({method_name}, {round(np.log10(CONV_COND))}, {C})")
                t0 = time.perf_counter()
                #===============================
                # VAR
                #===============================
                x_range = np.linspace(0, L, Nx)
                dx = x_range[1] - x_range[0]
                dt = np.min([CFL_dt(dx, D, C=C),0.95*dx])
                t_range = np.arange(0,Tmax, dt)

                XX,YY = np.meshgrid(x_range,x_range)
                ux, uy = import_field(Nx, "Lax_Wendroff", 0.05, 10**-2)
                ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]


                F, diffu_length, kmax, DIFFU_LENGTH_LIST, diffu_line = SIMU_MASS(METHOD)
                DIFFU_LENGTH.append(diffu_length)
                DIFFU_DERIV.append(np.max(np.abs(np.gradient(diffu_line,dx))))
                ax1.legend(ncol=7)
                line_diffu_length.set_data(Nx_range[:j+1], DIFFU_LENGTH) # /!\ DIFFU_LENGTH
                print(DIFFU_LENGTH[0])
                # ax2.plot(t_range[:len(MEAN_COND)]*1000, MEAN_COND)


                ax2.plot(t_range[:kmax+1]*1000,DIFFU_LENGTH_LIST) #, label = f"{method_name}"
                plt.pause(0.01)
                t1 = time.perf_counter()
                # print("stop:",kmax,"| T :",(t1-t0),"s")

                # def func(x, a, b,c):
                #     return -a * np.exp(-b * x) + c

                # popt, pcov = curve_fit(func, Nx_range[:j+1], np.array(DIFFU_DERIV),[3300,0.05,3300])
                # ax1.plot(Nx_range,func(Nx_range, *popt),"k--",label =f"{popt[-1]:.1f} +-{ np.sqrt(pcov[-1,-1]):.1f}")
                # ax1.set_label(f"{popt[-1]:.1f} +-{ np.sqrt(pcov[-1,-1]):.1f}")
            SAVE_ARR_LENGTH = np.vstack([SAVE_ARR_LENGTH, DIFFU_LENGTH])
            SAVE_ARR_DERIV = np.vstack([SAVE_ARR_DERIV, DIFFU_DERIV])
            ax1.legend(ncol=7)
            plt.pause(0.01)

SAVE = 0
if SAVE:
    directory = r"D:\\JLP\\CMI\\_MASTER 2_\\TC5-NUM\\AYMERIC\PROJECT 3\\MASS_PLOT\\"
    file_param = f"LENGTH_C{C}_INTERPOLATION.txt"
    headers_form = np.array(headers)[:,None]

    import pandas as pd
    df = pd.DataFrame(SAVE_ARR_LENGTH.T)
    df.columns = headers
    df.to_csv(directory+file_param, index=False, na_rep='NaN')

    file_param = f"DERIV_C{C}.txt"

    df2 = pd.DataFrame(SAVE_ARR_DERIV.T)
    df2.columns = headers
    df2.to_csv(directory+file_param, index=False, na_rep='NaN')