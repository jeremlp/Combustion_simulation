# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:49:11 2023

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

from WENO_PROJECT3 import RK3_mass_WENO, RK3_mass_LW, RK3_WENO_LW,RK3_LW_ONLY,WENO,RK3_LUD_Upwind,RK3_LUD_LW_FULL,RK3_WENO_LW_FULL
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp, odeint

plt.close("all")

def CFL_dt(dx, D, C):
    # Compute time step based on CFL condition
    dt_max = C*dx**2/D
    return dt_max

def solve_mass_temp_Upwind(F, T):
    for f in F:
        f[1:-1,1:-1] +=  Upwind(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
    T[1:-1,1:-1] += Upwind(T, ux_s, uy_s, dx,dt) + a*laplace(T,dx)*dt
    return F, T

def solve_mass_temp_LUD(F, T):
    for f in F:
        f[1:-1,1:-1] +=  Upwind(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
        f[2:-2,2:-2] += LUD(f, ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt) + D*laplace(f[1:-1,1:-1],dx)*dt
    T[1:-1,1:-1] += Upwind(T, ux_s, uy_s, dx,dt) + a*laplace(T,dx)*dt
    T[2:-2,2:-2] += LUD(T, ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt) + a*laplace(T[1:-1,1:-1],dx)*dt
    return F, T

def solve_mass_temp_LW(F, T):
    for f in F:
        f[1:-1,1:-1] +=  Lax_Wendroff(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
    T[1:-1,1:-1] += Lax_Wendroff(T, ux_s, uy_s, dx,dt) + a*laplace(T,dx)*dt
    return F, T

def solve_mass_temp_RK3_WENO(F,T):
    for f in F:
        f[1:-1,1:-1] = RK3_WENO_LW(f, ux, uy, dx,dt,D) #+ (f[1:-1,1:-1]*div(ux,uy,dx))[8:-8,8:-8]*dt
    T[1:-1,1:-1] = RK3_WENO_LW(T, ux, uy, dx,dt,a)
    # T[9:-9,9:-9] -=  (T[1:-1,1:-1]*div(ux,uy,dx))[8:-8,8:-8]*dt
    return F, T

def solve_mass_temp_RK3_LW(F,T):
    for f in F:
        f[1:-1,1:-1] = RK3_LW_ONLY(f, ux, uy, dx,dt,D)
    T[1:-1,1:-1] = RK3_LW_ONLY(T, ux, uy, dx,dt,a)
    return F, T

def solve_mass_temp_WENO(F,T):
    for f in F:
        f[1:-1,1:-1] += Lax_Wendroff(f, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(f,dx)*dt
        f[3:-3,3:-3] += WENO(f, ux, uy, dx,dt) + D*laplace(f[2:-2,2:-2],dx)*dt

    T[1:-1,1:-1] += Lax_Wendroff(T, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + a*laplace(T,dx)*dt
    T[3:-3,3:-3] += WENO(T, ux, uy, dx,dt) + a*laplace(T[2:-2,2:-2],dx)*dt
    return F, T

def solve_mass_temp_RK3_LUD(F,T):
    for f in F:
        f[1:-1,1:-1] = RK3_LUD_Upwind(f, ux, uy, dx,dt,D)
    T[1:-1,1:-1] = RK3_LUD_Upwind(T, ux, uy, dx,dt,D)
    return F, T
def solve_mass_temp_RK3_LUD_FULL(F,T):
    for f in F:
        f[1:-1,1:-1] = RK3_LUD_LW_FULL(f, ux, uy, dx,dt,D)
    T[1:-1,1:-1] = RK3_LUD_LW_FULL(T, ux, uy, dx,dt,D)
    return F, T

def solve_mass_RK3_WENO_FULL(F):
    for f in F:
        f[1:-1,1:-1] = RK3_WENO_LW_FULL(f, ux, uy, dx,dt,D)
    T[1:-1,1:-1] = RK3_WENO_LW_FULL(T, ux, uy, dx,dt,D)
    return F
"""
C = 0.1, chem_r = 250
LW : 2424.398 K

WENO : T = 2525.72 K

RK3 LW: T = 2424.00 K

RK3 WENO: T = 2450.83


FROM chem_r = 250 to 1000 -> no diff for LW (T=2506K, C=0.1)
FROM C = 0.1 to 0.05 -> T=2489 to 2479 K /!\ (chem_r = 250)

"""
@njit
def get_Q(F, T):
    f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F
    Q = A*(rho*f_CH4/W_CH4)*(rho*f_O2/W_O2)**2 * np.exp(-Ta/T)
    return Q

def dT(T,t, wT_dot):
    return wT_dot/(rho*cp)

@njit
def solve_chem(F, T, stoichio_coef, enthalpy, Wk):
    wT_dot = np.zeros((Nx, Nx))
    Q = get_Q(F,T)
    for k,f in enumerate(F):
        wk_dot = Q * Wk[k] * stoichio_coef[k]
        F[k] = f + wk_dot / rho * dtau
        wT_dot -= enthalpy[k]/Wk[k] * wk_dot

    T += wT_dot/(rho*cp) * dtau
    return F, T

def RK3_mass_LW(w, ux, uy, dx,dt,D):
    k1 = Lax_Wendroff(w,  ux[1:-1,1:-1], uy[1:-1,1:-1],dx,dt) #+ D*laplace(w,dx)*dt #48x48
    k2 = Lax_Wendroff(w[1:-1,1:-1] + k1/3, ux[2:-2,2:-2], uy[2:-2,2:-2],dx,dt) #+ D*laplace(w[1:-1,1:-1],dx)*dt #46x46
    k3 = Lax_Wendroff(w[2:-2,2:-2] + 2*k2/3, ux[3:-3,3:-3], uy[3:-3,3:-3],dx,dt) #+ D*laplace(w[2:-2,2:-2],dx)*dt #44x44
    return (k1[2:-2,2:-2]/4 + 3/4*k3)


def rhs(s, fT):
    # print(fT)
    Q = A*(rho*fT[2]/W_CH4)*(rho*fT[1]/W_O2)**2 * np.exp(-Ta/fT[-1])
    # print(np.max(Q))
    omega_k = np.array([Q*Wk[0]*stoichio_coef[0], Q*Wk[1]*stoichio_coef[1],Q*Wk[2]*stoichio_coef[2], Q*Wk[3]*stoichio_coef[3], Q*Wk[4]*stoichio_coef[4]])
    omega_T = - enthalpy[0]/Wk[0] * omega_k[0] - enthalpy[1]/Wk[1] * omega_k[1] - enthalpy[2]/Wk[2] * omega_k[2]  - enthalpy[3]/Wk[3] * omega_k[3] - enthalpy[4]/Wk[4] * omega_k[4]

    return np.hstack([omega_k/rho, omega_T/(rho*cp)])


def solve_chem_SCIPY(s, fT_ravel):
    fT = fT_ravel.reshape(6,Nx,Nx)
    Q = A*(rho*fT[2]/W_CH4)*(rho*fT[1]/W_O2)**2 * np.exp(-Ta/fT[-1])
    # print(np.max(Q))
    omega_k = np.array([Q*Wk[0]*stoichio_coef[0], Q*Wk[1]*stoichio_coef[1],Q*Wk[2]*stoichio_coef[2], Q*Wk[3]*stoichio_coef[3], Q*Wk[4]*stoichio_coef[4]])
    omega_T = - enthalpy[0]/Wk[0] * omega_k[0] - enthalpy[1]/Wk[1] * omega_k[1] - enthalpy[2]/Wk[2] * omega_k[2]  - enthalpy[3]/Wk[3] * omega_k[3] - enthalpy[4]/Wk[4] * omega_k[4]

    return np.concatenate([omega_k/rho, omega_T[None,:]/(rho*cp)]).ravel()

def solve_chem_implicit_for(F, T, Tmax, atol = 1e-3, rtol = 1e-6):
    F_res, T_res = np.copy(F), np.copy(T)
    for px in range(Nx):
        for py in range(Nx):
            # t0 = time.perf_counter()
            res = solve_ivp(rhs, (0, Tmax), [F[0][px,py], F[1][px,py], F[2][px,py], F[3][px,py], F[4][px,py], T[px,py]], atol=atol,rtol=rtol)
            F_res[0][px,py], F_res[1][px,py], F_res[2][px,py], F_res[3][px,py], F_res[4][px,py], T_res[px,py] = res.y[:,-1]
            # t1 = time.perf_counter()
            # time_ivp = (t1-t0)*1000
            # print((px,py), round(time_ivp,4),"ms")
    return F_res, T_res

def solve_chem_implicit_ivp(F,T,delta_t, atol = 1e-3, rtol = 1e-6):
    fT = np.concatenate([F,T[None,:]]).ravel()
    res_ravel = solve_ivp(solve_chem_SCIPY, (0, delta_t), fT, atol=atol,rtol=rtol)

    res = res_ravel.y[:,-1].reshape(6,Nx,Nx)

    return res[:-1], res[-1]


def solve_chem_implicit_ODE(F,T, delta_t, atol = 1e-3, rtol = 1e-6):
    fT = np.concatenate([F,T[None,:]]).ravel()
    res_ravel = odeint(solve_chem_SCIPY,fT,[0, delta_t],tfirst = True, atol=atol,rtol=rtol)

    res = res_ravel[-1].reshape(6,Nx,Nx)

    return res[:-1], res[-1]

def inject_mass(F):
    #f_N2, f_O2, f_CH4, f_H2O, f_CO2
    for i, f in enumerate(F):
        f[ 0, 1:Nx//4] =     [0.78, 0.22, 0, 0, 0][i] #AIR MIXTURE
        f[ 0, Nx//4:Nx//2] = [1,    0,    0, 0 ,0][i] #NITROGEN

        f[-1, 1:Nx//4] =     [0,    0,    1, 0, 0][i] #METHANE
        f[-1, Nx//4:Nx//2] = [1,    0,    0, 0, 0][i] #NITROGEN
    return F


def ignite(T):
    band = T[round(0.75/(1000*dx)):round(1.25/(1000*dx)),:]
    T[round(0.75/(1000*dx)):round(1.25/(1000*dx)),:] = np.maximum(band,np.full_like(band,T_ignite))
    return T

def set_F_T_boundary(F,T):
    F[:, :,0]  = F[:, :,1]
    F[:, :,-1]  = F[:, :,-2]
    F[:, 0,:]  = F[:, 1,:]
    F[:, -1,:] = F[:, -2,:]

    T[ :,0]  = T[ :,1]
    T[ :,-1]  = T[ :,-2]
    T[ 0,:]  = T[ 1,:]
    T[ -1,:] = T[ -2,:]
    return F, T
#==================
# CONSTANTS
#==================
D = 15*10**-6
nu = D #species diffusion
a = D #heat diffusion
L =  0.002 #2*10**-3
#               N2, O2, CH4, H2O, CO2
stoichio_coef = [0.0, -2.0, -1.0, 2.0, 1.0]
Wk = np.array([28.0134, 31.998, 16.04, 18.01528, 44.01])/1000 #molar mass kg/mol
W_O2, W_CH4 = Wk[1], Wk[2]
enthalpy = [0.0,0.0,-74.9*10**3, -241.818*10**3, -393.52*10**3] #J/mol

rho = 1.1614 #kg.m^-3
A = 1.1*10**8
Ta = 10_000 #K
cp = 1200 #J Kg^-1 K^-1

Tmax = 10*10**-3#10**-6

tau_diffu = L**2/D

x0, y0 = L/2,L/2

Nfluid = 5
#===============================================================================================

Nx = 40
C = 0.2
chem_time_ratio = 5000


ux, uy = import_field(Nx, "Lax_Wendroff", C=0.05, CONV_COND=10**-2)
ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]

max_vel = np.max(np.sqrt(ux**2+uy**2))

x_range = np.linspace(0, L, Nx)
dx = x_range[1] - x_range[0]
dt = np.min([CFL_dt(dx, D, C=C),0.95*dx/max_vel])
t_range = np.arange(0,Tmax, dt)

XX,YY = np.meshgrid(x_range,x_range)



F = np.zeros((Nfluid, Nx, Nx))
F[0,:,:] = 0.78
F[1,:,:] = 0.22 #FILL WITH AIR


f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F

T0 = 300.0
T_ignite = 1000.0

T = np.full((Nx, Nx), T0)

fig = plt.figure(layout="constrained",figsize=(10,7))
gs = fig.add_gridspec(3,3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[2, :])

ax5 = fig.add_subplot(gs[1, 0])
ax6 = fig.add_subplot(gs[1, 1])
ax7 = fig.add_subplot(gs[1, 2])


im_O2 = ax1.imshow(f_O2, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=0.25)
im_CH4 = ax3.imshow(f_CH4, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=1)
im_T = ax2.imshow(T, extent=[0,L*1000,0,L*1000], origin="lower", vmin=300,vmax=3200)

im_N2 = ax5.imshow(f_N2, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=1)
im_H2O = ax6.imshow(f_H2O, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=0.25)
im_CO2 = ax7.imshow(f_CO2, extent=[0,L*1000,0,L*1000], origin="lower", vmin=0,vmax=0.25)

plt.colorbar(im_O2, ax=ax1, pad = -0.1)
plt.colorbar(im_N2, ax=ax5, pad = -0.1)
plt.colorbar(im_T, ax=ax2, pad = -0.1)
plt.colorbar(im_H2O, ax=ax6, pad = -0.1)
plt.colorbar(im_CH4, ax=ax3, pad = -0.0)
plt.colorbar(im_CO2, ax=ax7, pad = -0.0)


temp_line,= ax4.plot([],[],"-")

ax1.set_title("O2")
ax2.set_title("T")
ax3.set_title("CH4")
ax5.set_title("N2")
ax6.set_title("H2O")
ax7.set_title("CO2")

ax4.set_xlabel("t (ms)")
ax4.set_ylabel("T (K)")
ax4.set_xlim(0,Tmax*1000)
ax4.set_ylim(250,3200)
ax4.grid()
fig.suptitle(f"T max = {np.max(T):.2f}")
plt.pause(0.01)
# fig.tight_layout()
# plt.pause(0.01)
# fig.tight_layout()
# plt.pause(0.01)

dtau = dt/chem_time_ratio
tau_range = np.arange(0,Tmax, dtau)

T_list = []
O2_list = []
CH4_list = []

is_ignite = False
ignite_count = 0
ti = time.perf_counter()

"""
ODE+IVP 203.13 s -> T 2427
ODE :  500 s -> T 2427
EULER: 780 -> T 2427 around

ALWAYS CHEM : 210 -> T 2427
"""
time_list = [0]
for k,t in enumerate(tqdm(t_range)):
    t0 = time.perf_counter()

    if t > 3*10**-3 and is_ignite == False:
        T = ignite(T)
        is_ignite = True
        # print(k,np.max(T),"================================")

    if t < np.inf:
        F = inject_mass(F)

    F, T = solve_mass_temp_LW(F, T)
    F, T = set_F_T_boundary(F,T)
    # print(np.min(F),np.max(F))
    F[F<0] = 0

    # if k==150: break
    if is_ignite:
        ignite_count += 1
        t0 = time.perf_counter()
        F, T = solve_chem_implicit_ivp(F,T, dt, atol = 1e-3, rtol = 1e-6)
        t1 = time.perf_counter()
        time_list.append((t1-t0)*1000)
        print((t1-t0)*1000,"ms", np.max(T))


    """t0 = time.perf_counter()
    print(np.max(solve_chem_implicit_ODE(F,T, dt, atol = 1e-1, rtol = 1e-6)[-1]))
    t1 = time.perf_counter()
    print((t1-t0)*1000,"ms")"""

    f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F
    # print(np.max(T)-np.max(T2))

    # print(np.max(T))
    T_list.append(np.max(T))
    O2_list.append(np.max(f_O2))
    CH4_list.append(np.max(f_CH4))

    # if k == 177: break
    if len(T_list) > 5:
        T_var = np.mean(np.diff(T_list[-5:]))

    if k%15==0 and k > 1:
        im_O2.set_data(f_O2)
        im_CH4.set_data(f_CH4)
        im_T.set_data(T)
        im_H2O.set_data(f_H2O)
        im_CO2.set_data(f_CO2)
        im_N2.set_data(f_N2)
        temp_line.set_data(t_range[:k+1]*1000, T_list)
        fig.suptitle(f"T max = {np.max(T):.2f}")
        plt.pause(0.01)
        save_dir = "D:\\JLP\CMI\\_MASTER 2_\\TC5-NUM\AYMERIC\\PROJECT 3\\ANIMATIONS\\CHEM\\"
        # plt.savefig(save_dir + "frame_" + f"{k}".zfill(5))
        # print(k)
    t1 = time.perf_counter()
    # print(k,round((t1-t0)*1000,2),"ms")

tf = time.perf_counter()
print("TOTAL TIME :", round(tf-ti,2),"s")