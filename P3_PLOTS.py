# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:32:34 2023

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
from WENO_PROJECT3 import RK3_mass_WENO, RK3_mass_LW, RK3_WENO_LW,RK3_LW_ONLY,WENO, RK3_LUD_Upwind,RK3_mass_LUD,RK3_LUD_LW

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.close("all")
def get_w_opt(Nx):
    return 2/(1+sin(pi/Nx))

def CFL_dt(dx, D, C):
    # Compute time step based on CFL condition
    dt_max = C*dx**2/D
    return dt_max


def solve_mass(f):
    ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]
    f[1:-1,1:-1] = f[1:-1,1:-1] + Upwind(f,ux, uy, ux_s, uy_s, dx,dt)
    f[1:-1,1:-1] += D*laplace(ux,dx)*dt
    return f

def RK3_LUD_FULL():
    pass #just to be recognized

def LUD_LW():
    pass

def solve_Navier(ux_in, uy_in,dx, dt, t, kmax, P_guess, METHOD):
    ux, uy = np.copy(ux_in), np.copy(uy_in)
    ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]
    ux_ss, uy_ss = ux[2:-2,2:-2], uy[2:-2,2:-2]
    ux_bis, uy_bis = np.zeros_like(ux_s),np.zeros_like(uy_s)
    if METHOD == LUD_LW or METHOD ==" LUD_Up":

        ux_bis[1:-1,1:-1] = ux_ss + LUD(ux,ux_ss, uy_ss, dx,dt) + nu*laplace(ux_s,dx)*dt
        uy_bis[1:-1,1:-1] = uy_ss + LUD(uy,ux_ss, uy_ss, dx,dt) + nu*laplace(uy_s,dx)*dt

        grid = np.arange(0,(Nx-2)**2).reshape(Nx-2,Nx-2)
        grid2 = np.copy(grid)
        grid2[1:-1,1:-1] = -1
        bord_mask = np.isin(grid, grid2)

        if METHOD==LUD_LW:
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
        ux_bis = ux[1:-1,1:-1] + Upwind(ux, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(ux,dx)*dt
        ux_bis[5:-5,5:-5] = ux[6:-6,6:-6] + RK3_mass_LUD(ux, ux, uy, dx,dt,D)   + D*laplace(ux[5:-5,5:-5],dx)*dt
        uy_bis = uy[1:-1,1:-1] + Upwind(uy, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(uy,dx)*dt
        uy_bis[5:-5,5:-5] = uy[6:-6,6:-6] + RK3_mass_LUD(uy, ux, uy, dx,dt,D)   + D*laplace(uy[5:-5,5:-5],dx)*dt
    elif METHOD == RK3_LUD_FULL:
        ux_bis =            ux[1:-1,1:-1] + Upwind(ux, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + nu*laplace(ux,dx)*dt
        ux_bis[1:-1,1:-1] = ux[2:-2,2:-2] + LUD(ux,ux_ss, uy_ss, dx,dt) + nu*laplace(ux_s,dx)*dt
        ux_bis[5:-5,5:-5] = ux[6:-6,6:-6] + RK3_mass_LUD(ux, ux, uy, dx,dt,D)   + nu*laplace(ux[5:-5,5:-5],dx)*dt

        uy_bis =            uy[1:-1,1:-1] + Upwind(uy, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + nu*laplace(uy,dx)*dt
        uy_bis[1:-1,1:-1] = uy[2:-2,2:-2] + LUD(uy,ux_ss, uy_ss, dx,dt) + nu*laplace(uy_s,dx)*dt
        uy_bis[5:-5,5:-5] = uy[6:-6,6:-6] + RK3_mass_LUD(uy, ux, uy, dx,dt,D)   + nu*laplace(uy[5:-5,5:-5],dx)*dt
    elif METHOD == RK3_LUD_LW:
        ux_bis = ux[1:-1,1:-1] + Lax_Wendroff(ux, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(ux,dx)*dt
        ux_bis[5:-5,5:-5] = ux[6:-6,6:-6] + RK3_mass_LUD(ux, ux, uy, dx,dt,D)   + D*laplace(ux[5:-5,5:-5],dx)*dt
        uy_bis = uy[1:-1,1:-1] + Lax_Wendroff(uy, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(uy,dx)*dt
        uy_bis[5:-5,5:-5] = uy[6:-6,6:-6] + RK3_mass_LUD(uy, ux, uy, dx,dt,D)   + D*laplace(uy[5:-5,5:-5],dx)*dt
    else:
        ux_bis = ux_s + METHOD(ux,ux_s, uy_s, dx,dt) + nu*laplace(ux,dx)*dt
        uy_bis = uy_s + METHOD(uy,ux_s, uy_s, dx,dt) + nu*laplace(uy,dx)*dt

        #
        #ux_bis = ux_s + METHOD(ux,ux_s, uy_s, dx,dt)
        #uy_bis = uy_s + METHOD(uy,ux_s, uy_s, dx,dt)
        #ux_bis += nu*laplace(uy,dx)*dt
        #uy_bis += nu*laplace(uy,dx)*dt

    #ELIPTIC -> P
    b = rdxo0/dt * div(ux,uy,dx)
    b_pad = np.pad(b, 1, mode="constant") #because div reduce size by 2 and we need b[i,j] -> 1, Nx-1

    # P_guess = np.zeros((Nx,Nx))
    # if t==0 or kmax > 25:
    """
    Nx 75; C=0.1
    tol = 0.0001 -> max strain: 1510.97, stop: 2363; 147.27s
    tol = 0.001 -> max strain: 1510.97, stop: 2363; 104s
    tol = 0.01 -> max strain: 1510.97; stop: 2363; 93s
    tol = 0.1 -> max strain: 1510.97; stop: 2363; 90.97s
    tol = 1 -> max strain: 1510.97; vel 1.12; norm 0.62; stop: 2363; 58s

    Nx 125; C=0.1
    tol = 1 -> max strain 1496.62; vel 1.15; norm 0.63

    Nx 200; C=0.1
    tol = 1 -> max strain 1491.68; vel 1.16; norm 0.64; stop 12472; 2973s
    """
    # P, ERROR, kmax = iteration_SOR(P_guess, b_pad, dx,Nx, w = get_w_opt(Nx), tolerence=1)
    P, ERROR, kmax = iteration_SOR(P_guess, b_pad, dx,Nx, w = get_w_opt(Nx), tolerence=0.1)

    # else:
    #     P, ERROR, kmax = iteration_SOR(P_guess, b_pad, dx,Nx, w = get_w_opt(Nx), tolerence=10)
    #     print("SOR:",kmax)
    #     P, ERROR, kmax = iteration_GS(P_guess, b_pad, dx,Nx, tolerence=10)
    #     print("GS:",kmax)

    # print("kmax:",kmax)
    if kmax == -1: #No convergence
        P, ERROR, kmax = iteration_SOR(np.zeros((Nx,Nx)), b_pad, dx,Nx, w = get_w_opt(Nx), tolerence=0.1)

    P[:,0] = P[:,1] #x =0
    P[:,-1] = 0 #P[:,-2] #x = L
    P[0,:] = P[1,:] #y = 0
    P[-1,:] = P[-2,:] #y = L

    #PRESSURE CORRECTION
    if METHOD == " LUD_LW" or METHOD ==" LUD_Up":
        new_ux = ux_bis - dt/rdxo0 *(deriv_x(P,dx))#[1:-1,1:-1] before for LUD
        new_uy = uy_bis - dt/rdxo0 *(deriv_y(P,dx))#[1:-1,1:-1]
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
    convergence_count = 0
    kmax = -1
    P_guess = np.zeros((Nx, Nx))

    # plt.legend()
    # plt.pause(0.001)
    for k,t in enumerate(tqdm(t_range)):
        t0 = time.perf_counter()
        if METHOD == " LUD_LW" or METHOD ==" LUD_Up":#(ux[2:-2,2:-2], uy[2:-2,2:-2])
            (ux[1:-1,1:-1], uy[1:-1,1:-1]), kmax, P_guess = solve_Navier(ux, uy, dx, dt, t, kmax, P_guess,METHOD)
        else:
            (ux[1:-1,1:-1], uy[1:-1,1:-1]), kmax, P_guess = solve_Navier(ux, uy, dx, dt, t, kmax, P_guess,METHOD)

        #EXIT CONDITIONS
        ux[:,-1] = ux[:,-2]
        uy[:,-1] = uy[:,-2]
        uy[:,0]  = uy[:,1]

        STRAIN_LIST.append(np.max(np.abs(deriv_y(uy, dx)[:,0])))
        EXIT_VEL_LIST.append(np.max(np.abs(ux[:,-1])))
        MEAN_NORM_LIST.append(np.mean(np.sqrt(ux[1:-1,1:-1]**2 + uy[1:-1,1:-1]**2)))
        # print(STRAIN_LIST[-1])
        if k > 2:
            cond_strain = np.abs((STRAIN_LIST[-1] - STRAIN_LIST[-2])/dt) / (STRAIN_LIST[-2]/tau_diffu)
            cond_exit_vel = np.abs((EXIT_VEL_LIST[-1] - EXIT_VEL_LIST[-2])/dt) / (EXIT_VEL_LIST[-2]/tau_diffu)
            cond_mean_norm = np.abs((MEAN_NORM_LIST[-1] - MEAN_NORM_LIST[-2])/dt) / (MEAN_NORM_LIST[-2]/tau_diffu)
            COND_STRAIN.append(cond_strain)
            COND_EXIT_VEL.append(cond_exit_vel)
            COND_MEAN_NORM.append(cond_mean_norm)

            if (cond_strain < CONV_COND) and (cond_exit_vel < CONV_COND) and (cond_mean_norm < CONV_COND):#10**-6:
                convergence_count +=1

                if convergence_count > 15:
                    # update_cond_plot(line,line2,line3, COND_MEAN_NORM, COND_EXIT_VEL, COND_STRAIN, STRAIN_LIST, EXIT_VEL_LIST, MEAN_NORM_LIST)

                    plt.pause(0.001)
                    print("max strain:",round(STRAIN_LIST[-1],8))
                    print("max vel exit:",round(EXIT_VEL_LIST[-1],2))
                    print("mean norm:",round(MEAN_NORM_LIST[-1],2))
                    return k,ux, uy, P_guess, STRAIN_LIST, COND_STRAIN,COND_EXIT_VEL,COND_MEAN_NORM, EXIT_VEL_LIST,MEAN_NORM_LIST
            else: convergence_count = 0
            # print(np.log10(cond_strain),np.log10(cond_exit_vel),np.log10(cond_mean_norm))

        if k%2==0 and k >2:
            pass
            # ax1.cla()
            # ax2.cla()
            # norm = np.sqrt(ux**2+uy**2)
            # ax1.quiver(XX,YY, ux/np.sqrt(norm), uy/np.sqrt(norm), norm)
            # # ax1.streamplot(XX,YY, ux, uy, density=0.5)
            # ax1.set_xlim(0,L)
            # ax1.set_ylim(0,L)
            # ax2.plot(np.arange(0,len(STRAIN_LIST),1),np.array(STRAIN_LIST))
            # plt.pause(0.01)
        # update_cond_plot(line,line2,line3, COND_MEAN_NORM, COND_EXIT_VEL, COND_STRAIN, STRAIN_LIST, EXIT_VEL_LIST, MEAN_NORM_LIST)
        t1 = time.perf_counter()
        # print(f"Time: {(t1-t0)*1000:.2f} ms")

def SAVE_FIELD(ux, uy):
    directory = "D:\\JLP\CMI\\_MASTER 2_\\TC5-NUM\AYMERIC\\PROJECT 3\\FIELDS_SAVE_V2\\"
    file_param = f"N{Nx}_{method_name}_C{C}_CONV{round(np.log10(CONV_COND))}_field.txt"
    np.savetxt(directory + "ux_"+file_param, ux)
    np.savetxt(directory + "uy_"+file_param, uy)
    print("==========================\n",file_param,"SAVED\n==========================")


#===============================
# PARAMETERS
#===============================

rdxo0 = 1.1614
Nx = 50
D = 15*10**-6
nu = D
L =  0.002 #2*10**-3

Tmax = 7*10**-3


tau_diffu = L**2/D

x0, y0 = L/2,L/2


"""
TEMPS CONVECTI : L/(2u)
TEMPS DIFFUSIOF: L^2/U

MAX des deux je compare

||du/dt|| / (ref/temps_ref)

LEFT WALL / SORTIE / AVERAGE
"""

Nx_range = np.arange(150,180,10) #[25, 50, 75, 100, 125]

CONV_COND = 10**-6

fig, (ax1,ax2) = plt.subplots(2)
ax1.set_xlabel('Nx')
ax1.set_ylabel('Convergence of\nquantities')

ax1.grid()
ax1.set_xlim(Nx_range[0]*0.9,Nx_range[-1]*1.1)
# ax1.set_ylim(0.5,1.8)
ax1.set_ylim(1350,1700)

ax2.set_xlabel('t (ms)')
ax2.set_ylabel('Average of conditions \n (Strain, exit vel, mean norm)')

ax2.grid()
ax2.set_xlim(0,30)
ax2.set_ylim(10**-8,10**3)
ax2.set_yscale("log")
ax2.axhline(CONV_COND, color = "k", ls ="--")
fig.tight_layout()
plt.pause(0.01)


SAVE_ARR = Nx_range
headers = ["Nx"]
for METHOD in [LUD_LW]: #, RK3_LW, " LUD_LW",Upwind
    method_name = str(METHOD).split(" ")[1]
    headers.append(method_name)
    for CONV_COND in [10**-2]:
        for C in [0.05]:
            line_strain, = ax1.plot([],[], ".-",label = f"strain/1000 ({method_name},{round(np.log10(CONV_COND))},{C})")
            line_fit, = ax1.plot([],[],"k--")
            # line_exit_vel, = ax1.plot([],[], ".-",label = f"exit vel ({method_name})")
            # line_mean_norm, = ax1.plot([],[],".-", label = f"mean norm ({method_name})")
            STRAIN_CONV = []
            EXIT_VEL_CONV = []
            MEAN_NORM_CONV = []
            for j,Nx in enumerate(Nx_range):
                print("====================")
                print(f"NX = {Nx} ({method_name}, {round(np.log10(CONV_COND))}, {C})")
                t0 = time.perf_counter()
                #===============================
                # VAR
                #===============================
                x_range = np.linspace(0, L, Nx)
                dx = x_range[1] - x_range[0]
                dt = np.min([CFL_dt(dx, D, C=C),0.95*dx])
                t_range = np.arange(0,Tmax, dt)

                XX,YY = np.meshgrid(x_range,x_range)
                ux_i,uy_i =  field(XX,YY)
                ux, uy = np.copy(ux_i), np.copy(uy_i)

                f0 = np.where((XX-x0)**2 + (YY-y0)**2 < (L/3)**2, 1., 0.)
                f = np.copy(f0)

                k_stop, ux, uy, P_guess, STRAIN_LIST, COND_STRAIN,COND_EXIT_VEL,COND_MEAN_NORM, EXIT_VEL_LIST,MEAN_NORM_LIST = SIMU(CONV_COND, METHOD)
                #===============================
                # SAVE FIELD
                #===============================
                # SAVE_FIELD(ux, uy)

                STRAIN_CONV.append(STRAIN_LIST[-1])
                EXIT_VEL_CONV.append(EXIT_VEL_LIST[-1])
                MEAN_NORM_CONV.append(MEAN_NORM_LIST[-1])

                line_strain.set_data(Nx_range[:j+1],np.array(STRAIN_CONV)) #/1000 for visualization
                # def func(x, a, b, c):
                #     return a * np.exp(-b * x) + c

                # if len(STRAIN_CONV) > 3:
                #     popt, pcov = curve_fit(func, Nx_range[:j+1], np.array(STRAIN_CONV),[1000,0.1,1490])
                #     line_fit.set_data(Nx_range,func(Nx_range, *popt))
                #     line_fit.set_label(f"{popt[-1]:.1f} +-{ np.sqrt(pcov[-1,-1]):.1f}")

                # line_exit_vel.set_data(Nx_range[:j+1],EXIT_VEL_CONV)
                # line_mean_norm.set_data(Nx_range[:j+1],MEAN_NORM_CONV)

                old_lines = ax2.get_lines() #Change alpha for previous lines
                for l in old_lines[1:]:
                    l.set_alpha(0.2)

                MEAN_COND = np.mean([COND_MEAN_NORM,COND_EXIT_VEL,COND_STRAIN],axis=0)
                ax2.plot(t_range[:k_stop-2]*1000,MEAN_COND)

                # simu_dict[Nx][C]["strain"] = STRAIN_LIST
                # simu_dict[Nx][C]["exit_vel"] = EXIT_VEL_LIST
                # simu_dict[Nx][C]["mean_norm"] = MEAN_NORM_LIST
                ax1.legend()
                plt.pause(0.001)
                t1 = time.perf_counter()



                print("stop:",k_stop,"| T :",(t1-t0),"s")

            SAVE_ARR = np.vstack([SAVE_ARR,STRAIN_CONV])

directory = r"D:\\JLP\\CMI\\_MASTER 2_\\TC5-NUM\\AYMERIC\PROJECT 3\\FIELD_PLOT\\"
file_param = f"{method_name}_CONV{round(np.log10(CONV_COND))}_C{C}.txt"
headers_form = np.array(headers)[:,None]

import pandas as pd
df = pd.DataFrame(SAVE_ARR.T)
df.columns = headers
df.to_csv(directory+file_param, index=False, na_rep='NaN')