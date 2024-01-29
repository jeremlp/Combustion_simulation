# -*- coding: utf-8 -*-
"""
Created on Mon Dec  16 2023

@author: jerem
"""
import numpy as np
from numba import njit
from P3_MODULES import deriv_x, deriv_y, laplace, div, Lax_Wendroff,Upwind,LUD, Centered ,import_field

@njit
def weno5(a, b, c, d, e):
    q1 = a / 3.0 - 7.0 / 6.0 * b + 11.0 / 6.0 * c
    q2 = -b / 6.0 + 5.0 / 6.0 * c + d / 3.0
    q3 = c / 3.0 + 5.0 / 6.0 * d - e / 6.0

    s1 = 13.0 / 12.0 * (a - 2.0 * b + c)**2 + 0.25 * (a - 4.0 * b + 3.0 * c)**2
    s2 = 13.0 / 12.0 * (b - 2.0 * c + d)**2 + 0.25 * (d - b)**2
    s3 = 13.0 / 12.0 * (c - 2.0 * d + e)**2 + 0.25 * (3.0 * c - 4.0 * d + e)**2

    eps = 1.0e-6
    a1 = 1.0 / (eps + s1)**2
    a2 = 6.0 / (eps + s2)**2
    a3 = 3.0 / (eps + s3)**2

    f_value = (a1 * q1 + a2 * q2 + a3 * q3) / (a1 + a2 + a3)
    return f_value
WenoVec = np.vectorize(weno5)

# @njit
def get_uR(w):
    uR = WenoVec(w[4:],w[3:-1],w[2:-2],w[1:-3],w[0:-4])
    return np.roll(np.array(uR),0)
# @njit
def get_uL(w):
    uL = WenoVec(w[0:-4],w[1:-3],w[2:-2],w[3:-1],w[4:])
    return np.roll(np.array(uL),1, axis=0)


def PDE_WENO_vy(w2, a2,dx):
    uL = get_uL(w2)
    uR = get_uR(w2)

    fR = a2[2:-2]*uR
    fL = a2[2:-2]*uL

    alpha = np.max([np.abs(a2[1:-1]), np.abs(a2[:-2])],0)[1:-1]
    F = 0.5*(fR + fL) - 0.5*alpha*(uR - uL)

    dF = -(F[2:] -F[1:-1])/dx

    return dF

def weno_x_a(w, ux, uy, dx):
    res = PDE_WENO_vy(w.T, ux.T,dx)
    return (res[:,3:-3]).T

def weno_y_a(w, ux, uy, dx):
    res = PDE_WENO_vy(w, uy,dx)
    return (res[:,3:-3])

def WENO(w, ux, uy, dx,dt):
    return (weno_y_a(w,ux,uy,dx) + weno_x_a(w,ux,uy,dx))*dt

def RK3_mass_LW(w, ux, uy, dx,dt,D):
    k1 = Lax_Wendroff(w,  ux[1:-1,1:-1], uy[1:-1,1:-1],dx,dt) #+ D*laplace(w,dx)*dt #48x48
    k2 = Lax_Wendroff(w[1:-1,1:-1] + k1/3, ux[2:-2,2:-2], uy[2:-2,2:-2],dx,dt) #+ D*laplace(w[1:-1,1:-1],dx)*dt #46x46
    k3 = Lax_Wendroff(w[2:-2,2:-2] + 2*k2/3, ux[3:-3,3:-3], uy[3:-3,3:-3],dx,dt) #+ D*laplace(w[2:-2,2:-2],dx)*dt #44x44
    return (k1[2:-2,2:-2]/4 + 3/4*k3)

def RK3_mass_LUD(w, ux, uy, dx,dt,D):
    k1 = LUD(w,  ux[2:-2,2:-2], uy[2:-2,2:-2],dx,dt)  #46x46
    k2 = LUD(w[2:-2,2:-2] + k1/3, ux[4:-4,4:-4], uy[4:-4,4:-4],dx,dt) #42x42
    k3 = LUD(w[4:-4,4:-4] + 2*k2/3, ux[6:-6,6:-6], uy[6:-6,6:-6],dx,dt) #38x38
    return (k1[4:-4,4:-4]/4 + 3/4*k3)

def RK3_mass_WENO(w, ux, uy, dx,dt,D):
    k1 = WENO(w,  ux, uy,dx,dt) #+ D*laplace(w[2:-2,2:-2],dx)*dt #44x44
    k2 = WENO(w[3:-3,3:-3] + k1/3, ux[3:-3,3:-3], uy[3:-3,3:-3],dx,dt) #+ D*laplace(w[5:-5,5:-5],dx)*dt #38x38
    k3 = WENO(w[6:-6,6:-6] + 2*k2/3, ux[6:-6,6:-6], uy[6:-6,6:-6],dx,dt) #+ D*laplace(w[8:-8,8:-8],dx)*dt #32x32
    return (k1[6:-6,6:-6]/4 + 3/4*k3)

def RK3_LW_ONLY(w, ux, uy, dx,dt,D):
    w[1:-1,1:-1] = w[1:-1,1:-1] + Lax_Wendroff(w, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(w,dx)*dt
    w[3:-3,3:-3] =  w[3:-3,3:-3] + RK3_mass_LW(w, ux, uy, dx,dt,D)   + D*laplace(w[2:-2,2:-2],dx)*dt
    return w[1:-1,1:-1]

def RK3_WENO_LW(w, ux, uy, dx,dt,D):
    w[1:-1,1:-1] = w[1:-1,1:-1] + Lax_Wendroff(w, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(w,dx)*dt
    w[3:-3,3:-3] =  w[3:-3,3:-3] + RK3_mass_LW(w, ux, uy, dx,dt,D)   + D*laplace(w[2:-2,2:-2],dx)*dt
    w[9:-9,9:-9] =  w[9:-9,9:-9] + RK3_mass_WENO(w, ux, uy, dx,dt,D) + D*laplace(w[8:-8,8:-8],dx)*dt
    return w[1:-1,1:-1]

def RK3_LUD_Upwind(w, ux, uy, dx,dt,D):
    w[1:-1,1:-1] = w[1:-1,1:-1] + Upwind(w, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(w,dx)*dt
    w[6:-6,6:-6] = w[6:-6,6:-6] + RK3_mass_LUD(w, ux, uy, dx,dt,D)   + D*laplace(w[5:-5,5:-5],dx)*dt
    return w[1:-1,1:-1]

def RK3_LUD_LW(w, ux, uy, dx,dt,D):
    w[1:-1,1:-1] = w[1:-1,1:-1] + Lax_Wendroff(w, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(w,dx)*dt
    w[6:-6,6:-6] = w[6:-6,6:-6] + RK3_mass_LUD(w, ux, uy, dx,dt,D)   + D*laplace(w[5:-5,5:-5],dx)*dt
    return w[1:-1,1:-1]

def RK3_LUD_LW_FULL(w, ux, uy, dx,dt,D):
    w[1:-1,1:-1] = w[1:-1,1:-1] + Lax_Wendroff(w, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(w,dx)*dt
    w[2:-2,2:-2] += LUD(w, ux[2:-2,2:-2], uy[2:-2,2:-2], dx,dt) + D*laplace(w[1:-1,1:-1],dx)*dt
    w[6:-6,6:-6] = w[6:-6,6:-6] + RK3_mass_LUD(w, ux, uy, dx,dt,D)   + D*laplace(w[5:-5,5:-5],dx)*dt
    return w[1:-1,1:-1]

def RK3_WENO_LW_FULL(w, ux, uy, dx,dt,D):
    w[1:-1,1:-1] = w[1:-1,1:-1] + Lax_Wendroff(w, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(w,dx)*dt
    w[3:-3,3:-3] =  w[3:-3,3:-3] + WENO(w, ux, uy, dx,dt) + D*laplace(w[2:-2,2:-2],dx)*dt
    w[9:-9,9:-9] =  w[9:-9,9:-9] + RK3_mass_WENO(w, ux, uy, dx,dt,D) + D*laplace(w[8:-8,8:-8],dx)*dt
    return w[1:-1,1:-1]


def set_boundary(w):
    w[:,0]  = w[:,1]
    w[:,-1]  = w[:,-2]
    w[0,:]  = w[1,:]
    w[-1,:] = w[-2,:]
    return w

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    plt.close("all")

    Nx = 70
    L =  0.002 #2*10**-3

    x = np.linspace(0,L,Nx)

    XX, YY = np.meshgrid(x,x)


    dx = x[1]- x[0]
    C = 0.1

    D = 15*10**-6

    dt = min(C*dx, C*dx**2/D)

    method_name = "Lax_Wendroff"
    ux, uy = import_field(Nx, method_name, C=0.05, CONV_COND=10**-2)

    # ux = ux*0
    # uy = uy*0

    ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]

    x0, y0 = L/3, L/2
    r2 = 4*(XX-x0)**2 + (YY-y0)**2
    w = np.where(r2 < (L/3)**2, 1., 0.)
    w = np.full((Nx,Nx),300)
    # w = np.sin(YY*np.pi)*np.sin(XX*np.pi)#+(np.random.random((Nx,Nx))-0.5)*0.11

    # a = w*0 + 1
    # plt.plot(x,w)
    # line, = plt.plot(x,w)
    im = plt.imshow(w,origin="lower", extent=[0,L,0,L])

    # norm = np.sqrt(ux**2+uy**2)
    # quiv = plt.quiver(XX,YY, L*ux/(np.sqrt(norm+0.0001)**2), L*uy/(np.sqrt(norm+0.0001)**2), norm) #•
    plt.colorbar()
    plt.pause(0.1)
    # plt.grid()

    Tmax = 0.005
    t_range = np.arange(0,Tmax,dt)


    plt.streamplot(XX, YY, ux, uy)
    old_max = 0
    speed_list = []
    for k,t in enumerate(t_range):

        t0 = time.perf_counter()


        w_LW, w_WENO,w_RK3_LW,w_RK3_WENO = np.copy(w), np.copy(w), np.copy(w), np.copy(w)

        ux_s, uy_s = ux[1:-1,1:-1], uy[1:-1,1:-1]

        # w_LW[1:-1,1:-1] = w[1:-1,1:-1] + Lax_Wendroff(w, ux[1:-1,1:-1], uy[1:-1,1:-1], dx,dt) + D*laplace(w,dx)*dt
        # w_RK3_LW[3:-3,3:-3] =  w[3:-3,3:-3] + RK3_mass_LW(w, ux, uy, dx,dt,D)   + D*laplace(w[2:-2,2:-2],dx)*dt
        # w_RK3_WENO[9:-9,9:-9] =  w[9:-9,9:-9] + RK3_mass_WENO(w, ux, uy, dx,dt,D) + D*laplace(w[8:-8,8:-8],dx)*dt #+

        # w[1:-1,1:-1] = w_LW[1:-1,1:-1]
        # w[3:-3,3:-3] = w_RK3_LW[3:-3,3:-3]
        # w[9:-9,9:-9] = w_RK3_WENO[9:-9,9:-9]
        w = RK3_WENO_LW(w, ux, uy, dx,dt,D)
        w = set_boundary(w)

        pos_max = np.where(w==np.max(w))[0][0]
        print(pos_max)
        speed_list.append(pos_max*dx)


        # w = np.apply_along_uxis(RK3, uxis=1, arr=w, dt=dt)
        t1 = time.perf_counter()
        print(round((t1-t0)*1000,3),"ms")
        if k%10==0:
            im.set_data(w)
            # line.set_ydata(w)
            # norm = np.sqrt(ux**2+uy**2)
            # quiv.set_UVC(ux/np.sqrt(norm+0.0001)**2,uy/np.sqrt(norm+0.0001)**2, norm)
            im.autoscale()
            plt.title(f"{round(t,4)} | ({np.min(w):.3f} , {np.max(w):.3f})")
            plt.pause(0.9)

    im.set_data(w)
    plt.pause(0.001)

    plt.figure()
    x = t_range[:k+1]
    y = speed_list
    a,b = np.polyfit(x[200:],speed_list[200:],1)
    plt.plot(x,speed_list)
    plt.plot(x[200:],a*x[200:]+b,"--", label = round(a,4))
    plt.legend()
    plt.grid()

    # plt.figure()
    # plt.imshow(-ux_s * deriv_x(w,dx),origin="lower")
    # plt.colorbar()
    # plt.title("LW x")

    # plt.figure()
    # plt.imshow(-uy_s * deriv_y(w,dx),origin="lower")
    # plt.colorbar()
    # plt.title("LW y")


    # plt.figure()
    # plt.imshow(weno_x_a(w,ux,uy,dx),origin="lower")
    # plt.colorbar()
    # plt.title("WENO x")

    # plt.figure()
    # plt.imshow(weno_y_a(w,ux,uy,dx),origin="lower")
    # plt.colorbar()
    # plt.title("WENO y")





