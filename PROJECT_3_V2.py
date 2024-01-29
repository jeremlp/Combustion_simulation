# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:32:24 2023

@author: jerem
"""

from importlib import reload

import math
import numpy as np
from numba import njit
import time
from numpy import sin, cos, pi, exp, sqrt

from tqdm import tqdm

from P3_MODULES import RK3_LW, Centered, LUD, Lax_Wendroff, Upwind, iteration_SOR,iteration_GS
from P3_MODULES import deriv_x, deriv_y, laplace, div
from WENO_PROJECT3 import RK3_mass_WENO, RK3_mass_LW, RK3_WENO_LW,RK3_LW_ONLY,WENO, RK3_LUD_Upwind,RK3_mass_LUD

import matplotlib.pyplot as plt
plt.close("all")
def get_w_opt(Nx):
    return 2/(1+sin(pi/Nx))

def CFL_dt(dx, D, C):
    # Compute time step based on CFL condition
    dt_max = C*dx**2/D
    return dt_max

def RK3_LUD_FULL():
    pass
def solve_mass(f):
    ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]
    f[1:-1,1:-1] = f[1:-1,1:-1] + Upwind(f,ux, uy, ux_s, uy_s, dx,dt)
    f[1:-1,1:-1] += D*laplace(ux,dx)*dt
    return f

def solve_Navier(ux_in, uy_in,dx, dt, t, kmax, P_guess, METHOD):
    ux, uy = np.copy(ux_in), np.copy(uy_in)
    ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]
    ux_ss, uy_ss = ux[2:-2,2:-2], uy[2:-2,2:-2]
    ux_bis = np.zeros_like(ux_s)
    uy_bis = np.zeros_like(uy_s)

    if METHOD == " LUD_LW" or METHOD ==" LUD_Up":

        ux_bis[1:-1,1:-1] = ux_ss + LUD(ux,ux_ss, uy_ss, dx,dt) + nu*laplace(ux_s,dx)*dt
        uy_bis[1:-1,1:-1] = uy_ss + LUD(uy,ux_ss, uy_ss, dx,dt) + nu*laplace(uy_s,dx)*dt

        grid = np.arange(0,(Nx-2)**2).reshape(Nx-2,Nx-2)
        grid2 = np.copy(grid)
        grid2[1:-1,1:-1] = -1
        bord_mask = np.isin(grid, grid2)

        if METHOD==" LUD_LW":
            ux_bis[bord_mask] = (ux_s + Lax_Wendroff(ux,ux_s, uy_s, dx,dt) + nu*laplace(ux,dx)*dt)[bord_mask]
            uy_bis[bord_mask] = (uy_s + Lax_Wendroff(uy,ux_s, uy_s, dx,dt) + nu*laplace(uy,dx)*dt)[bord_mask]
        else:
            ux_bis[bord_mask] = (ux_s + Upwind(ux,ux_s, uy_s, dx,dt) + nu*laplace(ux,dx)*dt)[bord_mask]
            uy_bis[bord_mask] = (uy_s + Upwind(uy,ux_s, uy_s, dx,dt) + nu*laplace(uy,dx)*dt)[bord_mask]

    elif METHOD == RK3_LW:
        ux_bis[2:-2,2:-2] = ux[3:-3,3:-3] + RK3_LW(ux,ux_s, uy_s, dx,dt) + nu*laplace(ux_ss,dx)*dt
        uy_bis[2:-2,2:-2] = uy[3:-3,3:-3] + RK3_LW(uy,ux_s, uy_s, dx,dt) + nu*laplace(uy_ss,dx)*dt

        grid = np.arange(0,(Nx-2)**2).reshape(Nx-2,Nx-2)
        grid2 = np.copy(grid)
        grid2[3:-3,3:-3] = -1 #remove already done
        bord_mask = np.isin(grid, grid2)
        ux_bis[bord_mask] = (ux_s + Lax_Wendroff(ux,ux_s, uy_s, dx,dt) + nu*laplace(ux,dx)*dt)[bord_mask]
        uy_bis[bord_mask] = (uy_s + Lax_Wendroff(uy,ux_s, uy_s, dx,dt) + nu*laplace(uy,dx)*dt)[bord_mask]

    elif METHOD == RK3_LUD_Upwind:
        ux_bis = ux[1:-1,1:-1] + Upwind(ux, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + nu*laplace(ux,dx)*dt
        ux_bis[5:-5,5:-5] = ux[6:-6,6:-6] + RK3_mass_LUD(ux, ux, uy, dx,dt,D)   + nu*laplace(ux[5:-5,5:-5],dx)*dt

        uy_bis = uy[1:-1,1:-1] + Upwind(uy, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + nu*laplace(uy,dx)*dt
        uy_bis[5:-5,5:-5] = uy[6:-6,6:-6] + RK3_mass_LUD(uy, ux, uy, dx,dt,D)   + nu*laplace(uy[5:-5,5:-5],dx)*dt
    elif METHOD == RK3_LUD_FULL:
        ux_bis =            ux[1:-1,1:-1] + Upwind(ux, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + nu*laplace(ux,dx)*dt
        ux_bis[1:-1,1:-1] = ux[2:-2,2:-2] + LUD(ux,ux_ss, uy_ss, dx,dt) + nu*laplace(ux_s,dx)*dt
        ux_bis[5:-5,5:-5] = ux[6:-6,6:-6] + RK3_mass_LUD(ux, ux, uy, dx,dt,D)   + nu*laplace(ux[5:-5,5:-5],dx)*dt

        uy_bis =            uy[1:-1,1:-1] + Upwind(uy, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + nu*laplace(uy,dx)*dt
        uy_bis[1:-1,1:-1] = uy[2:-2,2:-2] + LUD(uy,ux_ss, uy_ss, dx,dt) + nu*laplace(uy_s,dx)*dt
        uy_bis[5:-5,5:-5] = uy[6:-6,6:-6] + RK3_mass_LUD(uy, ux, uy, dx,dt,D)   + nu*laplace(uy[5:-5,5:-5],dx)*dt
    else:
        ux_bis = ux_s + METHOD(ux,ux_s, uy_s, dx,dt) + nu*laplace(ux,dx)*dt
        uy_bis = uy_s + METHOD(uy,ux_s, uy_s, dx,dt) + nu*laplace(uy,dx)*dt


    # uy_bis[0,:] = uy_bis[1,:]
    # uy_bis[-1,:] = uy_bis[-2,:]

    # ux_bis[:,-1] = ux_bis[:,-2]
    # ux_bis[:,0] = ux_bis[:,1]
    # uy_bis[:,-1] = uy_bis[:,-2]
    # uy_bis[:,0]  = uy_bis[:,1]

    # ux[1,:] = ux[2,:]
    # ux[-2,:] = ux[-3,:]

    #ELIPTIC -> P
    b = rdxo0/dt * div(ux,uy,dx)
    b_pad = np.pad(b, 1, mode="constant") #because div reduce size by 2 and we need b[i,j] -> 1, Nx-1


    P, ERROR, kmax = iteration_SOR(P_guess, b_pad, dx,Nx, w = get_w_opt(Nx), tolerence=0.001)

    P[:,0] = P[:,1] #x =0
    P[:,-1] = 0 #P[:,-2] #x = L
    P[0,:] = P[1,:] #y = 0
    P[-1,:] = P[-2,:] #y = L

    #PRESSURE CORRECTION
    if METHOD == LUD:
        new_ux = ux_bis - dt/rdxo0 *(deriv_x(P,dx))[1:-1,1:-1]
        new_uy = uy_bis - dt/rdxo0 *(deriv_y(P,dx))[1:-1,1:-1]
    else:
        new_ux = ux_bis - dt/rdxo0 *(deriv_x(P,dx))
        new_uy = uy_bis - dt/rdxo0 *(deriv_y(P,dx))
    return np.array([new_ux, new_uy]), kmax, P

def field(x,y):
    vx = 0*x
    vy = 0*y
    vy[0, 1:Nx//4] = 1
    vy[0, Nx//4:Nx//2] = 0.2

    vy[-1, 1:Nx//4] = -1
    vy[-1, Nx//4:Nx//2] = -0.2

    return vx, vy


def update_cond_plot(line,line2,line3, COND_MEAN_NORM, COND_EXIT_VEL, COND_STRAIN, STRAIN_LIST, EXIT_VEL_LIST, MEAN_NORM_LIST):
    # line.set_data(np.arange(0,len(COND_MEAN_NORM),1),COND_MEAN_NORM)
    # line2.set_data(np.arange(0,len(COND_EXIT_VEL),1),COND_EXIT_VEL)
    # line3.set_data(range(len(COND_STRAIN)),np.array(COND_STRAIN))

    line.set_data(np.arange(0,len(MEAN_NORM_LIST),1),MEAN_NORM_LIST)
    line2.set_data(np.arange(0,len(EXIT_VEL_LIST),1),EXIT_VEL_LIST)
    line3.set_data(np.arange(0,len(STRAIN_LIST),1),np.array(STRAIN_LIST)/1000)
    plt.pause(0.001)

def SIMU(CONV_COND, METHOD):
    DIV_U_LIST_MAX = []
    DIV_U_LIST_MEAN = []
    STRAIN_LIST = []
    COND_STRAIN = []
    EXIT_VEL_LIST = []
    COND_EXIT_VEL = []
    MEAN_NORM_LIST = []
    COND_MEAN_NORM = []
    TIME_RECORD = []
    convergence_count = 0
    kmax = -1
    P_guess = np.zeros((Nx, Nx))

    # plt.legend()
    # plt.pause(0.001)
    for k,t in enumerate(tqdm(t_range)):
        t0 = time.perf_counter()
        (ux[1:-1,1:-1], uy[1:-1,1:-1]), kmax, P_guess = solve_Navier(ux, uy, dx, dt, t, kmax, P_guess, METHOD)
        # (ux[2:-2,2:-2], uy[2:-2,2:-2]), kmax, P_guess = solve_Navier(ux, uy, dx, dt, t, kmax, P_guess,METHOD)



        #EXIT CONDITIONS
        ux[:,-1] = ux[:,-2]
        uy[:,-1] = uy[:,-2]
        uy[:,0]  = uy[:,1]

        STRAIN_LIST.append(np.max(np.abs(deriv_y(uy, dx)[:,0])))
        EXIT_VEL_LIST.append(np.max(np.abs(ux[:,-1])))
        MEAN_NORM_LIST.append(np.mean(np.sqrt(ux[1:-1,1:-1]**2 + uy[1:-1,1:-1]**2)))
        if k > 2:
            cond_strain = np.abs((STRAIN_LIST[-1] - STRAIN_LIST[-2])/dt) / (STRAIN_LIST[-2]/tau_diffu)
            cond_exit_vel = np.abs((EXIT_VEL_LIST[-1] - EXIT_VEL_LIST[-2])/dt) / (EXIT_VEL_LIST[-2]/tau_diffu)
            cond_mean_norm = np.abs((MEAN_NORM_LIST[-1] - MEAN_NORM_LIST[-2])/dt) / (MEAN_NORM_LIST[-2]/tau_diffu)
            COND_STRAIN.append(cond_strain)
            COND_EXIT_VEL.append(cond_exit_vel)
            COND_MEAN_NORM.append(cond_mean_norm)
            TIME_RECORD.append(t*1000)

            if (cond_strain < CONV_COND) and (cond_exit_vel < CONV_COND) and (cond_mean_norm < CONV_COND):#10**-6:
                convergence_count +=1

                if convergence_count > 15:
                    # update_cond_plot(line,line2,line3, COND_MEAN_NORM, COND_EXIT_VEL, COND_STRAIN, STRAIN_LIST, EXIT_VEL_LIST, MEAN_NORM_LIST)

                    plt.pause(0.001)
                    # print("max strain:",round(STRAIN_LIST[-1],2))
                    # print("max vel exit:",round(EXIT_VEL_LIST[-1],2))
                    # print("mean norm:",round(MEAN_NORM_LIST[-1],2))
                    return k,ux, uy, P_guess, STRAIN_LIST, COND_STRAIN,COND_EXIT_VEL,COND_MEAN_NORM, EXIT_VEL_LIST,MEAN_NORM_LIST
            else: convergence_count = 0
            # print(np.log10(cond_strain),np.log10(cond_exit_vel),np.log10(cond_mean_norm))

        if k%15==0:

            if STREAM_PLOT:
                ax1.cla()
                norm = np.sqrt(ux**2+uy**2)
                ax1.streamplot(XX*1000,YY*1000, ux, uy, density=1, color=norm)
                ax1.set_xlim(-0.025,L*1000+0.025)
                ax1.set_ylim(-0.025,L*1000+0.025)
                ax1.set_xlabel("X [mm]")
                ax1.set_ylabel("Y [mm]")
                line_strain_rate.set_data(TIME_RECORD,COND_STRAIN)
                line_mean_vel.set_data(TIME_RECORD,COND_MEAN_NORM)
                line_exit_vel.set_data(TIME_RECORD,COND_EXIT_VEL)
                ax1.plot([1,2],[0-0.02,0-0.02],"k-", lw=3)
                ax1.plot([1,2],[2+0.015,2+0.015],"k-", lw=3)
                ax1.plot([0-0.025,0-0.025],[0,2],"k-", lw=3)
                # print("=============================================")
                # print(len(t_range[:k+1]),len(COND_EXIT_VEL))
            else:
                # im.set_data(div(ux,uy,dx))
                norm = np.sqrt(ux**2+uy**2)
                quiv.set_UVC(ux/np.sqrt(norm+0.0001)**2,uy/np.sqrt(norm+0.0001)**2, norm)

            plt.pause(0.001)
            # save_dir = "D:\\JLP\CMI\\_MASTER 2_\\TC5-NUM\AYMERIC\\PROJECT 3\\ANIMATIONS\\FLOW\\"
            # plt.savefig(save_dir + "frame_" + f"{k}".zfill(5))
            # update_cond_plot(line,line2,line3, COND_MEAN_NORM, COND_EXIT_VEL, COND_STRAIN, STRAIN_LIST, EXIT_VEL_LIST, MEAN_NORM_LIST)
        t1 = time.perf_counter()
        # print("DIV U = ", round(np.max(np.abs(div(ux,uy,dx))),6))
        # print(k,round(t,5),"Time:",round((t1-t0)*1000,2),"ms")
        # print("Strain : ", round(STRAIN_LIST[-1]))
#===============================
# PARAMETERS
#===============================

rdxo0 = 1.1614
D = 15*10**-6
nu = D
L =  0.002 #2*10**-3

Tmax = 7*10**-3


tau_diffu = L**2/D

x0, y0 = L/2,L/2

simu_dict = {}


Nx_range = np.arange(20,80,10) #[25, 50, 75, 100, 125]


#======================================================
# PARAMETERS
#======================================================
METHOD = Upwind #" LUD_LW"
Nx = 50
C = 0.05
CONV_COND = 10**-1

STREAM_PLOT = True
#======================================================
# PARAMETERS
#======================================================

method_name = str(METHOD).split(" ")[1]
# line_strain, = ax1.plot([],[], ".-",label = f"strain/1000 ({method_name},{round(np.log10(CONV_COND))},{C})")
# line_exit_vel, = ax1.plot([],[], ".-",label = f"exit vel ({method_name})")
# line_mean_norm, = ax1.plot([],[],".-", label = f"mean norm ({method_name})")


STRAIN_CONV = []
EXIT_VEL_CONV = []
MEAN_NORM_CONV = []
print("====================")
print(f"NX = {Nx} ({method_name}, {round(np.log10(CONV_COND))}, {C})")
tf0 = time.perf_counter()
#===============================
# VARS
#===============================
x_range = np.linspace(0, L, Nx)
dx = x_range[1] - x_range[0]
dt = np.min([CFL_dt(dx, D, C=C),0.95*dx])
t_range = np.arange(0,Tmax, dt)

XX,YY = np.meshgrid(x_range,x_range)
ux_i,uy_i =  field(XX,YY)
ux, uy = np.copy(ux_i), np.copy(uy_i)



#======================================================
# MATPLOTLIB
#======================================================

norm = np.sqrt(ux**2+uy**2)

fig, (ax1,ax2) = plt.subplots(2, figsize=(8,6))
fig.suptitle(f"{method_name}\nNX = {Nx} ({round(np.log10(CONV_COND))}, {C})")

if STREAM_PLOT:
    ax1.streamplot(XX*1000,YY*1000, ux, uy, density=1, color=norm)
    ax1.set_xlim(-0.025,L*1000+0.025)
    ax1.set_ylim(-0.025,L*1000+0.025)
else:
    quiv = plt.quiver(XX,YY, ux/np.sqrt(norm), uy/np.sqrt(norm), norm)

ax1.set_xlabel("X [mm]")
ax1.set_ylabel("Y [mm]")

ax2.set_xlabel("t [ms]")
ax2.set_ylabel("Convergences of quantities")
ax2.set_yscale("log")
ax2.axhline(CONV_COND, color = "k", ls ="--")

line_strain_rate, =ax2.plot([],[], label = "max strain rate")
line_mean_vel, = ax2.plot([],[], label = "mean velocity norm")
line_exit_vel, = ax2.plot([],[], label = "max exit velocity")
ax2.legend()
ax2.grid()
ax2.set_xlim(0,Tmax*1000)
ax2.set_ylim(10**-4,10**6)
ax2.plot([0])
# fig.suptitle("Velocity field convergence with time")
fig.tight_layout()
plt.pause(0.001)
k_stop, ux, uy, P_guess, STRAIN_LIST, COND_STRAIN,COND_EXIT_VEL,COND_MEAN_NORM, EXIT_VEL_LIST,MEAN_NORM_LIST = SIMU(CONV_COND, METHOD)


tf1 = time.perf_counter()
print("TOTAL TIME:",k_stop,"| T :",(tf1-tf0),"s")
