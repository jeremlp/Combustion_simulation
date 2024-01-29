# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:27:06 2023

@autdxor: jerem
"""
import math

import numpy as np
from numpy import sin, cos, pi, exp, sqrt

import matplotlib.pyplot as plt

from numba import njit
import time

from tqdm import tqdm

from P3_MODULES import LUD, Lax_Wendroff, Upwind, iteration_SOR,iteration_GS
from P3_MODULES import deriv_x, deriv_y, laplace, div
from WENO_PROJECT3 import RK3_mass_WENO, RK3_mass_LW, RK3_WENO_LW,RK3_LW_ONLY,WENO, RK3_LUD_Upwind


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

def solve_Navier(ux_in, uy_in,dx, dt, t, kmax, P_guess):
    ux, uy = np.copy(ux_in), np.copy(uy_in)
    ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]
    ux_ss, uy_ss = ux[2:-2,2:-2], uy[2:-2,2:-2]
    ux_bis = np.zeros_like(ux_s)
    uy_bis = np.zeros_like(uy_s)

    METHOD = Lax_Wendroff
    if METHOD == LUD:
        ux_bis[1:-1,1:-1] = ux_ss + METHOD(ux,ux_ss, uy_ss, dx,dt) + nu*laplace(ux,dx)*dt
        uy_bis[1:-1,1:-1] = uy_ss + METHOD(uy,ux_ss, uy_ss, dx,dt) + nu*laplace(ux,dx)*dt

        ux_bis[:,0] = ux_bis[:,1] #x =0
        ux_bis[:,-1] = ux_bis[:,-2] #P[:,-2] #x = L
        ux_bis[0,:] = ux_bis[1,:] #y = 0
        ux_bis[-1,:] = ux_bis[-2,:] #y = L

    else:
        ux_bis = ux_s + METHOD(ux,ux_s, uy_s, dx,dt) + nu*laplace(ux,dx)*dt
        uy_bis = uy_s + METHOD(uy,ux_s, uy_s, dx,dt) + nu*laplace(ux,dx)*dt


    #ELIPTIC -> P
    b = rdxo0/dt * div(ux,uy,dx)
    b_pad = np.pad(b, 1, mode="constant") #because div reduce size by 2 and we need b[i,j] -> 1, Nx-1


    P, ERROR, kmax = iteration_SOR(P_guess, b_pad, dx,Nx, w = get_w_opt(Nx), tolerence=0.1)

    P[:,0] = P[:,1] #x =0
    P[:,-1] = 0 #P[:,-2] #x = L
    P[0,:] = P[1,:] #y = 0
    P[-1,:] = P[-2,:] #y = L

    #PRESSURE CORRECTION
    new_ux = ux_bis - dt/rdxo0 *(deriv_x(P,dx))
    new_uy = uy_bis - dt/rdxo0 *(deriv_y(P,dx))
    return np.array([new_ux, new_uy]), kmax, P

rdxo0 = 1.1614
Nx = 70
D = 15*10**-6
nu = D
L =  0.002 #2*10**-3

Tmax = 0.002+1

tau_diffu = L**2/D

x_range = np.linspace(0, L, Nx)
dx = x_range[1] - x_range[0]
print("dt max", 0.95*dx, CFL_dt(dx, D, C=0.05))
dt = np.min([CFL_dt(dx, D, C=0.05),0.95*dx]) #dx*0.95 #CFL_dt(dx, D, C=0.001) #0.001

XX,YY = np.meshgrid(x_range,x_range)

t_range = np.arange(0,Tmax, dt)


CONV_COND = 10**-1

def field(x,y):
    vx = 0*x
    vy = 0*y
    print(Nx//20,Nx//4)
    vy[0, 1:Nx//4] = 1
    vy[0, Nx//4:Nx//2] = 0.2

    vy[-1, 1:Nx//4] = -1
    vy[-1, Nx//4:Nx//2] = -0.2
    return vx, vy

def N2(x,y):
    n2 = x*0
    n2[0,1:Nx//4] = 1


ux_i,uy_i =  field(XX,YY)


ux, uy = np.copy(ux_i), np.copy(uy_i)
norm = np.sqrt(ux**2+uy**2)

x0, y0 = L/2,L/2
f0 = np.where((XX-x0)**2 + (YY-y0)**2 < (L/3)**2, 1., 0.)
f = np.copy(f0)
P_guess = np.zeros((Nx, Nx))

kmax = -1
convergence_count = 0
#=================================
# VECTOR FIELD
#=================================
fig, ax = plt.subplots()
# stream = ax.streamplot(XX,YY,ux, uy, density=0.5)

quiv = plt.quiver(XX,YY, ux/np.sqrt(norm), uy/np.sqrt(norm), norm)
# im = plt.imshow(deriv_x(P_guess,dx), origin="lower", extent=[0,L,0,L], cmap = "gray",vmin = -100,vmax=100)
plt.colorbar()
plt.xlabel("x (cm)")
#=================================
# QUANTITIES
#=================================
# fig_im, ax_im = plt.subplots()
# im = ax_im.imshow(f0, origin="lower", extent=[0,L,0,L])
# cb = fig_im.colorbar(im)


plt.pause(0.01)
DIV_U_LIST_MAX = []
DIV_U_LIST_MEAN = []
STRAIN_LIST = []
COND_STRAIN = []
EXIT_VEL_LIST = []
COND_EXIT_VEL = []
MEAN_NORM_LIST = []
COND_MEAN_NORM = []
for k,t in enumerate(t_range):
    t0 = time.perf_counter()
    (ux[1:-1,1:-1], uy[1:-1,1:-1]), kmax, P_guess = solve_Navier(ux, uy, dx, dt, t, kmax, P_guess)


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

        if (cond_strain < CONV_COND) and (cond_exit_vel < CONV_COND) and (cond_mean_norm < CONV_COND):#10**-6:
            convergence_count +=1

            if convergence_count > 15:
                # update_cond_plot(line,line2,line3, COND_MEAN_NORM, COND_EXIT_VEL, COND_STRAIN, STRAIN_LIST, EXIT_VEL_LIST, MEAN_NORM_LIST)

                plt.pause(0.001)
                print("max strain:",round(STRAIN_LIST[-1],2))
                print("max vel exit:",round(EXIT_VEL_LIST[-1],2))
                print("mean norm:",round(MEAN_NORM_LIST[-1],2))

        # f = solve_mass(f)
        # f[:,0]  = f[:,1]
        # f[0,:]  = f[1,:]
        # f[-1,:] = f[-2,:]

    if k%10==0:
        norm = np.sqrt(ux**2+uy**2)
        # plt.cla()
        # ax.streamplot(XX,YY,ux, uy, color=norm, density=1)
        ax.set_xlim(0,L)
        ax.set_ylim(0,L)
        # ax.plot([0,0.25*L],[0,0], c="red", lw=3)
        # ax.plot([0.25*L,0.5*L],[0,0], c="blue", lw=3)

        # ax.plot([0,0.25*L],[L*0.995,L*0.995], c="red", lw=3)
        # ax.plot([0.25*L,0.5*L],[L*0.995,L*0.995], c="blue", lw=3)

        # im = plt.imshow(deriv_x(P_guess,dx), origin="lower", extent=[0,L,0,L], cmap = "gray", vmin=-100, vmax=100)
        # plt.colorbar()
        quiv.set_UVC(ux/np.sqrt(norm+0.0001)**2,uy/np.sqrt(norm+0.0001)**2, norm)
        # im.set_data(deriv_x(P_guess,dx))
        # im.autoscale()

        plt.pause(0.001)
    t1 = time.perf_counter()
    print("DIV U = ", round(np.max(np.abs(div(ux,uy,dx))),6))
    print(k,round(t,5),"Time:",round((t1-t0)*1000,2),"ms")
    print("Strain : ", round(STRAIN_LIST[-1]))

plt.figure()
plt.plot(DIV_U_LIST_MAX,label = "div MAX")
plt.plot(DIV_U_LIST_MEAN,label = "div MEAN")
plt.legend()
plt.grid()
plt.xlabel("k")
plt.ylabel("div")

"""
for species -> grad Phi = 0 at wall
grad P = 0 at wall & inlet

BUT P imposed at exit
"""


"""
Maximum strain rate  :
Nx = 125 : 8063 (C=0.1 : 8541) (C=0.02 : 7542)
Nx = 100 : 6474
Nx = 75 : 4870 (C=0.01 : 4370)
"""

def plot_P():
    plt.figure()
    plt.imshow(P_guess, origin = "lower", extent = [0,L*10,0,L*10])
    plt.title("P")
    plt.colorbar()
    plt.xlabel("x (cm)")

def plot_dP():
    plt.figure()
    plt.imshow(deriv_x(P_guess,dx), origin = "lower", extent = [0,L*10,0,L*10])
    plt.title("dP/dx")
    plt.colorbar()
    plt.xlabel("x (cm)")

def plot_div():
    plt.figure()
    plt.plot(DIV_U_LIST_MAX,label = "div MAX")
    # plt.plot(DIV_U_LIST_MEAN,label = "div MEAN")
    plt.legend()
    plt.grid()
    plt.xlabel("k")
    plt.ylabel("div")

