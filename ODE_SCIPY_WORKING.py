# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:56:05 2023

@author: jerem
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp,odeint
from numba import njit
plt.close("all")
#               N2, O2, CH4, H2O, CO2
stoichio_coef = [0.0, -2.0, -1.0, 2.0, 1.0]
Wk = np.array([28.0134, 31.998, 16.04, 18.01528, 44.01])/1000 #molar mass kg/mol
enthalpy = [0.0,0.0,-74.9*10**3, -241.818*10**3, -393.52*10**3] #J/mol

A = 1.1*10**8
rho = 1.1614 #kg.m^-3
cp = 1200 #J Kg^-1 K^-1

Ta = 10000
W_O2, W_CH4 = Wk[1], Wk[2]
#O2, CH4, N2
# @njit

# def rhs(s, fT):
#     fT = fT.reshape(6,Nx,Nx)
#     Q = A*(rho*fT[2]/W_CH4)*(rho*fT[1]/W_O2)**2 * np.exp(-Ta/fT[-1])
#     df = [Q*Wk[0]*stoichio_coef[0], Q*Wk[1]*stoichio_coef[1],Q*Wk[2]*stoichio_coef[2],Q*Wk[3]*stoichio_coef[3],  Q*Wk[4]*stoichio_coef[4]]
#     wT_dot = - enthalpy[0]/Wk[0] * df[0] - enthalpy[1]/Wk[1] * df[1] - enthalpy[2]/Wk[2] * df[2] - enthalpy[3]/Wk[3] * df[3] - enthalpy[4]/Wk[4] * df[4]

#     return np.array([df + [wT_dot/(rho*cp)]]).ravel()


# def apply_solve(F, T):
#     for px in range(Nx):
#         for py in range(Nx):
#             res = solve_ivp(rhs, (0, dt), [F[0][px,py], F[1][px,py], F[2][px,py], F[3][px,py], F[4][px,py], T[px,py]])
#             F[0][px,py], F[1][px,py], F[2][px,py], F[3][px,py], F[4][px,py], T[px,py] = res.y[:,-1]
#     return F, T

# def apply_solve_2D(F,T):

#     fT = np.concatenate([F,T[None,:]]).ravel()
#     res_ravel = solve_ivp(rhs, (0, dt), fT)

#     res = res_ravel.y[:,-1].reshape(6,Nx,Nx)

#     return res[:-1], res[-1]

apply_solve_2D_vec = np.vectorize(apply_solve_2D)

# def apply_solve_2D_ode(a,b,c,d,e,temp):
#     # print(a,b,c,d,e,temp)
#     res = odeint(rhs, [a,b,c,d,e,temp],(0, dt),tfirst = True)
#     return (res[1][0],res[1][1],res[1][2],res[1][3],res[1][4],res[1][-1])


def get_Q(F, T):
    f_N2, f_O2, f_CH4, f_H2O, f_CO2 = F
    Q = A*(rho*f_CH4/W_CH4)*(rho*f_O2/W_O2)**2 * np.exp(-Ta/T)
    return Q

def solve_chem(F, T, dtau):
    Q = get_Q(F,T)
    wT_dot = np.zeros((Nx, Nx))
    for k,f in enumerate(F):
        wk_dot = Q * Wk[k] * stoichio_coef[k]
        F[k] = f + wk_dot / rho * dtau
        wT_dot -= enthalpy[k]/Wk[k] * wk_dot
    T += wT_dot/(rho*cp) * dtau
    return F, T

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

def solve_chem_implicit_for(F, T, Tmax):
    print(F,T)
    F_res, T_res = np.copy(F), np.copy(T)
    for px in range(Nx):
        for py in range(Nx):
            res = solve_ivp(rhs, (0, Tmax), [F[0][px,py], F[1][px,py], F[2][px,py], F[3][px,py], F[4][px,py], T[px,py]])
            F_res[0][px,py], F_res[1][px,py], F_res[2][px,py], F_res[3][px,py], F_res[4][px,py], T_res[px,py] = res.y[:,-1]
    return F_res, T_res

def solve_chem_implicit(F,T, delta_t, method="RK45"):
    fT = np.concatenate([F,T[None,:]]).ravel()
    res_ravel = solve_ivp(solve_chem_SCIPY, (0, delta_t), fT, method=method)

    res = res_ravel.y[:,-1].reshape(6,Nx,Nx)

    return res[:-1], res[-1]

def solve_chem_implicit_ODE(F,T, delta_t):
    fT = np.concatenate([F,T[None,:]]).ravel()
    res_ravel = odeint(solve_chem_SCIPY,fT,[0, delta_t],tfirst = True)

    res = res_ravel[-1].reshape(6,Nx,Nx)

    return res[:-1], res[-1]



"""
Max temp (25,17) 3138 K or 2837 K

F[:,25,17] = array([0.73748561, 0.20695184, 0.05556254, 0.        , 0.        ])
T = 1000.0
"""


Nx = 1
dt =  7e-7#2.720810801618883e-06
chem_r = 1000
dtau = dt/chem_r

#N2, O2, CH4, H2O, CO2

fig, (ax1,ax2) = plt.subplots(2)

FT = np.array([0.73748561, 0.20695184, 0.05556254, 0. , 0., 1000 ])

FT = np.array([0.69336419, 0.18400766, 0.12262816, 0.        , 0.        ,1000]) #crash

t_range = np.arange(0,dt,dt/chem_r)
res_odeint = odeint(rhs,FT,t_range,tfirst = True)
res_ivp = solve_ivp(rhs, (0, dt), FT)

ax1.plot(res_ivp.t*1000, res_ivp.y.T[:,-1],"o-", color = "k", label = "Scipy")

ax2.plot(res_ivp.t*1000, res_ivp.y.T[:,1:-1],"-", color = "k")




# ax2.plot(res_ivp.t*1000, res_ivp.y.T[:,1:-1],"o", color = "k")
# ax1.plot(t_range*1000, res_odeint[:,-1],"k-")

F3,T3 = np.copy(FT[:-1]),FT[-1]

list_O2 = []
list_CH4 = []
list_H2O = []
list_CO2 = []
list_T= []

for t in t_range:
      F3,T3 = solve_chem(F3,T3,dtau)
      list_O2.append(F3[1])
      list_CH4.append(F3[2])
      list_H2O.append(F3[3])
      list_CO2.append(F3[4])
      list_T.append(T3[0][0])

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = ["blue", "orange", "green","red", "green"]
every = 25
ax1.plot(t_range[::every]*1000,list_T[::every],"x", label = "T Euler",color = colors[0])
ax2.plot(t_range[::every]*1000,list_O2[::every],"x", label = "O2 Euler", color = colors[0])
ax2.plot(t_range[::every]*1000,list_CH4[::every],"x", label = "CH4", color = colors[1])
ax2.plot(t_range[::every]*1000,list_H2O[::every],"x", label = "H2O", color = colors[2])
ax2.plot(t_range[::every]*1000,list_CO2[::every],"x", label = "CO2", color = colors[3])

# ax1.xlim(0,)
# plt.plot(res.t, res.y.T[:,-1],".-")
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend(loc='upper right', ncol = 5)

ax2.set_xlabel("t (ms)")
ax2.set_ylabel("Species Yk (%)")
ax1.set_ylabel("Temperature (K)")

fig.suptitle(f"Euler integration vs scipy integration (dt = 7e-07, chem_r = {chem_r})")

print("Euler:", T3[0][0])
print("Odeint:", res_odeint[-1,-1])
print("delta:", np.abs(res_odeint[-1,-1]-T3[0][0])/res_odeint[-1,-1]*100,"%")



t0 = time.perf_counter()
res_odeint = odeint(rhs,FT,t_range,tfirst = True, rtol =1e-2 , atol =1e-4)
t1 = time.perf_counter()
time_ode = (t1 - t0) *1000

t0 = time.perf_counter()
res_ivp = solve_ivp(rhs, (0, dt), FT)
t1 = time.perf_counter()

time_ivp = (t1 - t0) *1000

Error_list = []
Time_list = []
chem_range = np.arange(100,1000,10)
for chem_r in chem_range:
    dtau =  dt/chem_r
    t_range = np.arange(0,dt,dtau)

    # FT = np.array([0.73748561, 0.20695184, 0.05556254, 0. , 0., 1000 ])
    FT = np.array([0.69336419, 0.18400766, 0.12262816, 0.        , 0.        ,1000]) #crash

    F3,T3 = np.copy(FT[:-1]),FT[-1]
    t0 = time.perf_counter()
    for t in t_range:
          F3,T3 = solve_chem(F3,T3, dtau)
    t1 = time.perf_counter()
    Error_list.append(T3[0][0])
    Time_list.append((t1-t0)*1000)
    print(chem_r,"delta:", np.round(np.abs(res_ivp.y[-1,-1]-T3[0][0])/res_ivp.y[-1,-1]*100,5),"%")


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# plt.plot(np.arange(200,5000,50),np.abs(np.array(Error_list)-res_odeint[-1,-1])*100/res_odeint[-1,-1])
ax1.plot(chem_range,np.abs(np.array(Error_list)-res_ivp.y[-1,-1])*100/res_ivp.y[-1,-1], label = "$\Delta T$ (%)")
ax2.plot(chem_range,np.array(Time_list) / time_ivp, color = 'tab:red', label = "Compute time ratio")

ax1.set_xlabel("dt/dtau")
ax1.set_ylabel("Error $\Delta T$ (%)")
ax1.grid()
ax2.legend()
ax1.legend()
plt.title("Error $\Delta T$ (%) Euler & Scipy integration,\nfor different ratio dt/dtau")

ax2.set_ylabel("Compute time ratio t_euler / t_ivp")

ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0,16)