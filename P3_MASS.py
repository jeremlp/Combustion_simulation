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
from WENO_PROJECT3 import RK3_mass_WENO, RK3_mass_LW, RK3_WENO_LW,RK3_LW_ONLY,WENO,RK3_LUD_LW_FULL,RK3_WENO_LW_FULL,RK3_WENO_LW_FULL

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


#==================
# FIELD PARAMETERS
#==================
method_name = "Lax_Wendroff"
Nx = 40
C = 0.15
CONV_COND = 10**-2

#==================
# CONSTANTS
#==================
rdxo0 = 1.1614
D = 15*10**-6
nu = D
L =  0.002 #2*10**-3

Tmax = 10*10**-3

tau_diffu = L**2/D

x0, y0 = L/2,L/2

Nfluid = 5
#===============================================================================================
x_range = np.linspace(0, L, Nx)
dx = x_range[1] - x_range[0]
dt = np.min([CFL_dt(dx, D, C=C),0.95*dx])
t_range = np.arange(0,Tmax, dt)

XX,YY = np.meshgrid(x_range,x_range)

ux, uy = import_field(Nx, "Lax_Wendroff", C=0.05, CONV_COND=10**-2)
ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]


norm = sqrt(ux**2+uy**2)
# plt.quiver(XX, YY, ux, uy, norm)
# plt.streamplot(XX, YY, ux, uy, color = norm,density = 0.25, broken_streamlines = False)



F = np.zeros((Nfluid, Nx, Nx))
F[0,:,:] = 0.78
F[1,:,:] = 0.22 #FILL WITH AIR
f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F

fig = plt.figure(layout="constrained",figsize=(9,6))
gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])


ax1.streamplot(XX*1000, YY*1000, ux, uy, color = norm,density = 0.25, broken_streamlines = False, cmap = "gray")
ax2.streamplot(XX*1000, YY*1000, ux, uy, color = norm,density = 0.25, broken_streamlines = False, cmap = "gray")
ax3.streamplot(XX*1000, YY*1000, ux, uy, color = norm,density = 0.25, broken_streamlines = False, cmap = "gray")

im_N2 = ax1.imshow(f_N2, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=1)
im_O2 = ax2.imshow(f_O2, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=1)

im_CH4 = ax3.imshow(f_CH4, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=1)
fig.colorbar(im_CH4)


line, = ax4.plot(x_range*1000,f_N2[:,0])
ax4.set_ylim(0,1.1)
ax4.grid()
ax1.set_title("N2")
ax2.set_title("O2")
ax3.set_title("CH4")
ax1.set_xlabel("x[mm]")
ax1.set_ylabel("y[mm]")

ax4.set_xlabel("y[mm]")
ax4.set_ylabel("N2 (%)")

fig.suptitle("Species advection")
ax1.set_xlim(-0.025,L*1000+0.025)
ax1.set_ylim(-0.025,L*1000+0.025)
ax2.set_xlim(-0.025,L*1000+0.025)
ax2.set_ylim(-0.025,L*1000+0.025)
ax3.set_xlim(-0.025,L*1000+0.025)
ax3.set_ylim(-0.025,L*1000+0.025)
# ax2.set_ylim(0,1.1)
# ax2.grid()
# fig.tight_layout()
plt.pause(0.01)

METHOD = Lax_Wendroff

EXIT_MASS_LIST, MEAN_MASS_LIST, MID_DERIV_LIST = [], [], []

DIFFU_LENGTH_LIST = []

conv_count = 0
for k,t in enumerate(tqdm(t_range)):
    # print(round(t,5))
    t0 = time.perf_counter()
    F = inject_mass(F)
    F = solve_mass_Upwind(F)
    F[:, :,0]  = F[:, :,1]
    F[:, :,-1]  = F[:, :,-2]
    F[:, 0,:]  = F[:, 1,:]
    F[:, -1,:] = F[:, -2,:]
    f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F

    # print(np.min(f_N2),np.max(f_N2))


    #INTERPOLATION:
    y = f_N2[:,0]
    x_mm = x_range*1000
    order = y.argsort()
    y_data = y[order]
    x_data = x_mm[order]
    diffu_length = np.interp(0.1*np.max(y), y_data, x_data,  left=None, right=None, period=None)-np.interp(0.9*np.max(y), y_data, x_data,  left=None, right=None, period=None)

    diffu_zone = np.where((f_N2[:,0] < 0.9*np.max(f_N2[:,0])) & (f_N2[:,0] > 0.1*np.max(f_N2[:,0])))[0]
    line.set_ydata(f_N2[:,0])
    if len(diffu_zone) > 1:
        # diffu_length = np.abs(x_range[diffu_zone[-1]] - x_range[diffu_zone[0]])
        print(round(diffu_length,4),"mm")
        DIFFU_LENGTH_LIST.append(diffu_length)

        if t > 3*10**-3 and DIFFU_LENGTH_LIST[-1] == DIFFU_LENGTH_LIST[-2]:
            conv_count +=1
            print(t*1000,"ms |",conv_count, "|",MID_DERIV_LIST[-1])
            if conv_count > 500:
                break
        else:
            conv_count = 0

    EXIT_MASS_LIST.append(np.max(F[:, :,-1]))
    MEAN_MASS_LIST.append(np.mean(f_N2[1:-1,1:-1]))
    a = np.gradient(f_N2[:,0],dx)
    MID_DERIV_LIST.append(np.where(a == np.min(a))[0][0])
    # print(MID_DERIV_LIST[-1])


    if k%5==0 and k > 5:
        im_N2.set_data(f_N2)
        im_CH4.set_data(f_CH4)
        im_O2.set_data(f_O2)
        fig.suptitle(f"Lax_Wendroff (0.1) | t = {t*1000:.2f}ms")
        plt.pause(0.001)
        # save_dir = "D:\\JLP\CMI\\_MASTER 2_\\TC5-NUM\AYMERIC\\PROJECT 3\\ANIMATIONS\\MASS\\"
        # plt.savefig(save_dir + "frame_" + f"{k}".zfill(5))
    t1 = time.perf_counter()
    # print(round(np.max(f[1:-1,1:-1]),3), round(np.min(f[1:-1,1:-1]),3))
    # print((t1-t0)*1000,"ms")
plt.figure()
A = np.interp(0.1*np.max(y), y_data, x_data,  left=None, right=None, period=None)
B = np.interp(0.9*np.max(y), y_data, x_data,  left=None, right=None, period=None)
plt.plot(x_mm, y,".-")
plt.plot(x_mm[diffu_zone], y[diffu_zone],"-")
plt.axhline(0.1*np.max(y), color = "k", ls="--")
plt.axhline(0.9*np.max(y), color = "k", ls="--")
plt.grid()
plt.axvline(A, color = "r", ls="--")
plt.axvline(B, color = "r", ls="--")