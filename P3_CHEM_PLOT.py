# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 00:39:03 2023

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

from scipy.integrate import odeint,solve_ivp

plt.close("all")

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

def inject_mass(F):
    #f_N2, f_O2, f_CH4, f_H2O, f_CO2
    for i, f in enumerate(F):
        f[ 0, 1:Nx//4] =     [0.78, 0.22, 0, 0, 0][i] #AIR MIXTURE
        f[ 0, Nx//4:Nx//2] = [1,    0,    0, 0 ,0][i] #NITROGEN
        f[-1, 1:Nx//4] =     [0,    0,    1, 0, 0][i] #METHANE
        f[-1, Nx//4:Nx//2] = [1,    0,    0, 0, 0][i] #NITROGEN
    return F

# @njit
def get_Q(F, T):
    f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F
    Q = A*(rho*f_CH4/W_CH4)*(rho*f_O2/W_O2)**2 * np.exp(-Ta/T)
    return Q

# @njit
def solve_chem(F,T, stoichio_coef, enthalpy, Wk):
    Q = get_Q(F,T)
    wT_dot = np.zeros((Nx, Nx))
    for k,f in enumerate(F):
        wk_dot = Q * Wk[k] * stoichio_coef[k]
        F[k] = f + wk_dot / rho * dtau
        wT_dot -= enthalpy[k]/Wk[k] * wk_dot
    T += wT_dot/(rho*cp) * dtau
    return F, T

def solve_chem_SCIPY(s, fT_ravel):
    fT = fT_ravel.reshape(6,Nx,Nx)
    Q = A*(rho*fT[2]/W_CH4)*(rho*fT[1]/W_O2)**2 * np.exp(-Ta/fT[-1])
    # print(np.max(Q))
    omega_k = np.array([Q*Wk[0]*stoichio_coef[0], Q*Wk[1]*stoichio_coef[1],Q*Wk[2]*stoichio_coef[2], Q*Wk[3]*stoichio_coef[3], Q*Wk[4]*stoichio_coef[4]])
    omega_T = - enthalpy[0]/Wk[0] * omega_k[0] - enthalpy[1]/Wk[1] * omega_k[1] - enthalpy[2]/Wk[2] * omega_k[2]  - enthalpy[3]/Wk[3] * omega_k[3] - enthalpy[4]/Wk[4] * omega_k[4]

    return np.concatenate([omega_k/rho, omega_T[None,:]/(rho*cp)]).ravel()
def rhs(s, fT):
    # print(fT)
    Q = A*(rho*fT[2]/W_CH4)*(rho*fT[1]/W_O2)**2 * np.exp(-Ta/fT[-1])
    # print(np.max(Q))
    omega_k = np.array([Q*Wk[0]*stoichio_coef[0], Q*Wk[1]*stoichio_coef[1],Q*Wk[2]*stoichio_coef[2], Q*Wk[3]*stoichio_coef[3], Q*Wk[4]*stoichio_coef[4]])
    omega_T = - enthalpy[0]/Wk[0] * omega_k[0] - enthalpy[1]/Wk[1] * omega_k[1] - enthalpy[2]/Wk[2] * omega_k[2]  - enthalpy[3]/Wk[3] * omega_k[3] - enthalpy[4]/Wk[4] * omega_k[4]

    return np.hstack([omega_k/rho, omega_T/(rho*cp)])

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

    return res[:-1], res[-1]
def solve_mass_temp_Upwind(F, T):
    for f in F:
        f[1:-1,1:-1] +=  Upwind(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
    T[1:-1,1:-1] += Upwind(T, ux_s, uy_s, dx,dt) + a*laplace(T,dx)*dt
    return F, T

def solve_mass_temp_LUD_Up(F, T):
    for f in F:
        f[1:-1,1:-1] +=  Upwind(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
        f[2:-2,2:-2] += LUD(f, ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt) + D*laplace(f[1:-1,1:-1],dx)*dt
    T[1:-1,1:-1] += Upwind(T, ux_s, uy_s, dx,dt) + a*laplace(T,dx)*dt
    T[2:-2,2:-2] += LUD(T, ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt) + a*laplace(T[1:-1,1:-1],dx)*dt
    return F, T
def solve_mass_temp_LUD_LW(F, T):
    for f in F:
        f[1:-1,1:-1] +=  Lax_Wendroff(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
        f[2:-2,2:-2] += LUD(f, ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt) + D*laplace(f[1:-1,1:-1],dx)*dt
    T[1:-1,1:-1] += Lax_Wendroff(T, ux_s, uy_s, dx,dt) + a*laplace(T,dx)*dt
    T[2:-2,2:-2] += LUD(T, ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt) + a*laplace(T[1:-1,1:-1],dx)*dt
    return F, T

def solve_mass_temp_LW(F, T):
    for f in F:
        f[1:-1,1:-1] +=  Lax_Wendroff(f, ux_s, uy_s, dx,dt) + D*laplace(f,dx)*dt
    T[1:-1,1:-1] += Lax_Wendroff(T, ux_s, uy_s, dx,dt) + a*laplace(T,dx)*dt
    return F, T
def solve_mass_temp_RK3_WENO(F,T):
    for f in F:
        f[1:-1,1:-1] = RK3_WENO_LW(f, ux, uy, dx,dt,D)
    T[1:-1,1:-1] = RK3_WENO_LW(T, ux, uy, dx,dt,a)
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

def solve_mass_temp_RK3_WENO_FULL(F, T):
    for f in F:
        f[1:-1,1:-1] = RK3_WENO_LW_FULL(f, ux, uy, dx,dt,D)
    T[1:-1,1:-1] = RK3_WENO_LW_FULL(T, ux, uy, dx,dt,D)
    return F, T

def SIMU_CHEM(SOLVE_MASS_TEMP_FUNC, SOLVE_CHEM_FUNC, F_in,T_in):
    F, T = np.copy(F_in), np.copy(T_in)
    T_list = []
    convergence_count = 0
    is_ignite = False
    ignite_count = 0
    for k,t in enumerate(tqdm(t_range)):
        if t > 3*10**-3 and is_ignite == False:
            T = ignite(T)
            is_ignite = True

        t0 = time.perf_counter()
        F = inject_mass(F)

        F, T = SOLVE_MASS_TEMP_FUNC(F, T)
        F,T = set_F_T_boundary(F,T)
        F[F<0] = 0
        #===============================
        # SOLVE CHEM
        #===============================
        # if k == 173: break
        # print(k)
        if is_ignite:
            ignite_count += 1
            t0 = time.perf_counter()
            F, T = solve_chem_implicit_ivp(F,T, dt, atol = 1e-3, rtol = 1e-6)
            t1 = time.perf_counter()
            # print((t1-t0)*1000,"ms", np.max(T))

            # if ignite_count < 10:
            #     for _ in range(chem_time_ratio):
            #         F, T = solve_chem(F, T , stoichio_coef, enthalpy, Wk)
            # elif t > 3.25*10**-3 and t < 3.7*10**-3:
            #     print('ode')
            #     F, T = solve_chem_implicit_ODE(F,T, dt)
            # else:
            #     F, T = solve_chem_implicit_ivp(F,T, dt)


        # if SOLVE_CHEM_FUNC == solve_chem_implicit_ODE or SOLVE_CHEM_FUNC == solve_chem_implicit_ivp:
        #     if is_ignite:
        #         ignite_count += 1
        #         if ignite_count < 25:
        #             if Nx > 60:
        #                 # print("============== ODE ==============")
        #                 F, T = solve_chem_implicit_ODE(F,T, dt)
        #             elif Nx < 60:
        #                 # print("============== IVP ==============")
        #                 F, T = solve_chem_implicit_ivp(F,T, dt)
        #             else:
        #                 # print("============== CHEM ==============")
        #                 for _ in range(chem_time_ratio):
        #                     F, T = solve_chem(F, T , stoichio_coef, enthalpy, Wk)
        #         else:
        #             F, T = solve_chem_implicit_ivp(F,T, dt)


        # print(Nx,np.max(T))
        f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F
        T_list.append(np.max(T))
        t1 = time.perf_counter()

    return np.max(T), k,T_list


#==============================================
# CONSTANTS
#==============================================
rho = 1.1614 #kg.m^-3
A = 1.1*10**8
Ta = 10_000 #K
cp = 1200 #J Kg^-1 K^-1

D = 15*10**-6
nu = D #species diffusion
a = D #heat diffusion
L =  0.002 #2*10**-3

stoichio_coef = [0., -2., -1., 2., 1.]
Wk = np.array([28.0134, 31.998, 16.04, 18.01528, 44.01])/1000 #molar mass kg/mol
W_O2, W_CH4 = Wk[1], Wk[2]

enthalpy = [0.,0.,-74.9*10**3, -241.818*10**3, -393.52*10**3] #J/mol
T0 = 300.0
T_ignite = 1000.0

Nfluid = 5


#==============================================
# PLOT PARAMETERS
#==============================================
Nx_range = np.arange(40,70,10)
# chem_time_ratio = 250



Tmax = 10*10**-3 #s

fig, (ax1,ax2) = plt.subplots(2)
ax1.grid()
ax1.set_ylim(2400,2465)
ax1.set_xlim(Nx_range[0]*0.9,Nx_range[-1]*1.1)
ax1.set_xlabel("Nx")
ax1.set_ylabel("Tmax (K)")

ax2.grid()
ax2.set_ylim(2200,3300)
ax2.set_xlim(0,Tmax*1000)
ax2.set_xlabel("t (ms)")
ax2.set_ylabel("T (K)")
plt.pause(0.01)

#TEMP
NAME_FILE = 'CHEM_PLOT/C0.1_Nx140_10_Upwind_LW.txt'
NAME_SAVE = NAME_FILE[:-3] + "png"
import pandas as pd
pf = pd.read_csv(NAME_FILE)
arr = pf.to_numpy().T

for k,l in enumerate(arr[1:]):
    l = l.astype(float)
    Nx = arr[0].astype(float)
    mask = ~np.isnan(l)

    ax1.plot(arr[0],l,".--", label = pf.columns[k+1])

ax1.legend()
plt.pause(0.01)
SAVE_ARR = Nx_range

#solve_mass_temp_LUD_LW !
# METHOD_LIST = [solve_mass_temp_Upwind,solve_mass_temp_LW,
#                solve_mass_temp_WENO, solve_mass_temp_RK3_LW,
#                solve_mass_temp_LUD_LW, solve_mass_temp_WENO_LW,
#                solve_mass_temp_RK3_WENO]
CHEM_METHOD_LIST = [solve_chem_implicit_ivp]
headers = ["Nx"]
#Upwind,LW,RK3_LW,RK3_LUD_FULL,RK3_WENO_FULL,LUD
for SOLVE_MASS_TEMP_FUNC in [solve_mass_temp_LW]:
    method_name = str(SOLVE_MASS_TEMP_FUNC).split(" ")[1][16:]
    headers.append(method_name)
    for SOLVE_CHEM_FUNC in CHEM_METHOD_LIST:
        for C in [0.2]:
            for chem_time_ratio in [5000]:
                line, = ax1.plot([],[], ".-", label = f"{method_name} ({C})")
                T_conv = []
                print("=============",method_name,"==============")
                for j,Nx in enumerate(Nx_range):

                    x_range = np.linspace(0, L, Nx)
                    dx = x_range[1] - x_range[0]
                    dt = np.min([CFL_dt(dx, D, C=C),0.95*dx])
                    dtau = dt/chem_time_ratio
                    t_range = np.arange(0,Tmax, dt)

                    XX,YY = np.meshgrid(x_range,x_range)

                    ux, uy = import_field(Nx, "Lax_Wendroff", C=0.05, CONV_COND=10**-2)
                    ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]

                    F = np.zeros((Nfluid, Nx, Nx))
                    F[0,:,:] = 0.78
                    F[1,:,:] = 0.22 #FILL WITH AIR

                    T = np.full((Nx, Nx), T0)
                    # T = ignite(T)
                    T_max, kmax, T_list = SIMU_CHEM(SOLVE_MASS_TEMP_FUNC, SOLVE_CHEM_FUNC, F,T)
                    T_conv.append(T_max)
                    line.set_data(Nx_range[:j+1],T_conv)
                    print(Nx,T_max,"\n")
                    plt.pause(0.01)

                    ax1.legend()
                    plt.pause(0.01)
                    ax2.plot(t_range*1000, T_list)
                SAVE_ARR = np.vstack([SAVE_ARR,T_conv])
                plt.pause(0.01)
SAVE = 0
if SAVE == 1:
    directory = r"D:\\_MASTER1_\\CMI\\_MASTER 2_\\PROJECT 3\\CHEM_PLOT\\"

    file_param = f"C{C}_chem_r{chem_time_ratio}_Nx110_20.txt"
    headers_form = np.array(headers)[:,None]

    import pandas as pd
    df = pd.DataFrame(SAVE_ARR.T)
    df.columns = headers
    df.to_csv(directory+file_param, index=False, na_rep='NaN')


