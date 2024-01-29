# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:12:04 2023

@author: jerem
"""
import numpy as np
from numba import njit

def pad(fcalc):
    # Add ghost cells for periodic conditions (Numpy method)
    return np.pad(fcalc, 1, mode="wrap")

@njit(parallel=True)
def deriv_y(f,dx):
    return (f[2:, 1:-1] - f[:-2, 1:-1])/(2*dx)
@njit(parallel=True)
def deriv_x(f,dx):
    return (f[1:-1, 2:] - f[1:-1, :-2])/(2*dx)
@njit(parallel=True)
def laplace(f,dx):
    return (f[1:-1, :-2] + f[:-2, 1:-1] -4*f[1:-1, 1:-1] + f[1:-1, 2:] + f[2:, 1:-1])/(dx**2)

@njit(parallel=True)
def div(ux,uy,dx):
    return deriv_x(ux,dx) + deriv_y(uy,dx)


@njit(parallel=True)
def deriv_x_backward_O2(fpad_pad, h):#-48
    return - (-fpad_pad[2:-2,0:-4]+ 4*fpad_pad[2:-2,1:-3] -3*fpad_pad[2:-2,2:-2])/(2*h)
@njit(parallel=True)
def deriv_x_forward_O2(fpad_pad, h):#0
    return - (3*fpad_pad[2:-2,2:-2]-4*fpad_pad[2:-2,3:-1]+fpad_pad[2:-2,4:])/(2*h)

@njit(parallel=True)
def deriv_y_backward_O2(fpad_pad, h):#0
    return - (-fpad_pad[0:-4,2:-2]+ 4*fpad_pad[1:-3,2:-2] -3*fpad_pad[2:-2,2:-2])/(2*h)
@njit(parallel=True)
def deriv_y_forward_O2(fpad_pad, h):
    return - (3*fpad_pad[2:-2,2:-2]-4*fpad_pad[3:-1,2:-2]+fpad_pad[4:,2:-2])/(2*h)

@njit(parallel=True)
def deriv_x_forward(f,h):
    return (f[1:-1,2:] - f[1:-1,1:-1])/h
@njit(parallel=True)
def deriv_y_forward(f,h):
    return (f[2:,1:-1] - f[1:-1,1:-1])/h
@njit(parallel=True)
def deriv_x_backward(f,h):
    return (f[1:-1,1:-1] - f[1:-1,:-2])/h
@njit(parallel=True)
def deriv_y_backward(f,h):
    return (f[1:-1,1:-1] - f[:-2,1:-1])/h

@njit
def LUD(u,ux_ss, uy_ss, dx,dt):
    Du = np.where(ux_ss < 0, -ux_ss * deriv_x_forward_O2(u, dx), -ux_ss * deriv_x_backward_O2(u, dx))
    Du += np.where(uy_ss < 0, -uy_ss * deriv_y_forward_O2(u, dx), -uy_ss * deriv_y_backward_O2(u, dx))
    return Du*dt

@njit
def Upwind(u,ux_s, uy_s, dx,dt):
    Du = np.where(ux_s < 0, -ux_s * deriv_x_forward(u,dx), -ux_s * deriv_x_backward(u,dx))
    Du += np.where(uy_s < 0, -uy_s * deriv_y_forward(u,dx), -uy_s * deriv_y_backward(u,dx))
    return Du*dt

@njit
def Centered(u,ux_s, uy_s, dx,dt):
    return dt*(-ux_s*deriv_x(u,dx) - uy_s * deriv_y(u,dx))

@njit
def Lax_Wendroff(u,ux_s, uy_s, dx,dt):

    eps_x = ux_s*dt/dx
    eps_y = uy_s*dt/dx

    corr_x = 0.5*eps_x**2 * (u[1:-1, :-2] -2*u[1:-1, 1:-1] + u[1:-1, 2:])
    corr_y = 0.5*eps_y**2 * (u[:-2, 1:-1] -2*u[1:-1, 1:-1] + u[2:, 1:-1])

    corr_tdxree = 0.25*eps_x*eps_y*(u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] +u[:-2, :-2])
    corr_four = -1/8*(eps_x**2+eps_y**2)*(u[2:, 2:] - 2*u[2:, 1:-1] + u[2:, :-2]
                                        -2*u[1:-1, 2:] + 4*u[1:-1, 1:-1] - 2*u[1:-1, :-2]
                                        + u[:-2, 2:] -2*u[:-2, 1:-1] + u[:-2, :-2])
    Du = (-ux_s * deriv_x(u,dx) - uy_s * deriv_y(u,dx))*dt + corr_x + corr_y - corr_tdxree - corr_four
    return Du


def RK3_LW(u,ux_s, uy_s, dx,dt):
    k1 = Lax_Wendroff(u, ux_s, uy_s, dx,dt)
    k2 = Lax_Wendroff(u[1:-1,1:-1] + k1/3,   ux_s[1:-1,1:-1], uy_s[1:-1,1:-1], dx,dt)
    k3 = Lax_Wendroff(u[2:-2,2:-2] + 2*k2/3, ux_s[2:-2,2:-2], uy_s[2:-2,2:-2], dx,dt)
    return (k1[2:-2,2:-2]/4 + 3/4*k3)


@njit
def Gauss_Seidel(f, b, dx, Nx):
    res = np.copy(f)
    for i in range(1,Nx-1):
        for j in range(1,Nx-1):
            res[i,j] = 0.25*(f[i+1,j] + f[i,j+1] + res[i-1,j] + res[i,j-1] - dx**2* b[i,j])
    return res
@njit
def iteration_GS(P_guess, b, dx,Nx, tolerence):
    phi = np.copy(P_guess)
    ERROR = []
    for k in range(10000):
        phi = Gauss_Seidel(phi, b, dx, Nx)
        error_max = np.max(np.abs(    laplace(phi,dx) - b[1:-1,1:-1]))
        ERROR.append(error_max)
        if error_max < tolerence:
            return phi, ERROR, k
    return phi, ERROR, -1


@njit
def SOR(f, b, dx,Nx, w):
    res = np.copy(f)
    for i in range(1,Nx-1):
        for j in range(1,Nx-1):
            res[i,j] = (1-w)*f[i,j] + w*0.25*(f[i+1,j] + f[i,j+1] + res[i-1,j] + res[i,j-1] - dx**2* b[i,j])
    return res

@njit
def iteration_SOR(P_guess, b, dx,Nx, w, tolerence):
    phi = np.copy(P_guess)
    ERROR = []
    for k in range(10000):
        phi = SOR(phi, b, dx, Nx, w)
        error_max = np.max(np.abs(    laplace(phi,dx) - b[1:-1,1:-1]))
        ERROR.append(error_max)
        if error_max < tolerence:
            return phi, ERROR, k
    return phi, ERROR, -1


def import_field(Nx, method_name, C, CONV_COND):
    directory =  "FIELDS_SAVE_V2\\"
    file_param = f"N{Nx}_{method_name}_C{C}_CONV{round(np.log10(CONV_COND))}_field.txt"
    ux = np.loadtxt(directory + "ux_"+file_param)
    uy = np.loadtxt(directory + "uy_"+file_param)
    return ux, uy

def CFL_dt(dx, D, C):
    # Compute time step based on CFL condition
    dt_max = C*dx**2/D
    return dt_max