from numba import jit, njit, vectorize
from numba import float64
import numpy as np
import matplotlib.pyplot as plt
import computational as comp
import stability_check as stcheck

#compute force on top
#-----------------------------------------------------------------------------#

#eval. external field
def B_external(t, omega, amp):
    return amp*np.cos(omega*t)*np.array([0,1,0])

#eval. magnetic force 
def g(r,u,m,g):
    u=np.array(u)
    mag_force=comp.gradient_point(r[0],r[1],r[2],0.0001,F,u)
    return mag_force/m-g*np.array([0,0,1])

#eval. base magnet field at given point [rx,ry,rz]
def F(rx,ry,rz,u):
    b=comp.Get_B_point(rx,ry,rz,kx,ky,kz,x,y,z,d) 
    return np.dot(u,b)

#eval. rigt side of system of eqs. 2.8
def Fv(f_i, I, It, m_0, m, ga, fric, alpha, t, amp_ext, omega_ext):
    r   = f_i[:3]
    v   = f_i[3:6]
    k   = f_i[6:9]
    W_n = f_i[9:12]
    w_z = f_i[12]
    n   = f_i[13:16]

    k = k / np.linalg.norm(k)
    n = n - np.dot(n, k) * k
    n = n / np.linalg.norm(n)

    omega_vec = W_n + w_z * k

    # magnetic moment mu
    m_vec = -m_0 * (np.cos(alpha) * k + np.sin(alpha) * n)

    drdt = v
    dvdt = g(r, m_vec, m, ga)

    dkdt = np.cross(W_n, k)
    dndt = np.cross(omega_vec, n)

    b = comp.Get_B_point(r[0], r[1], r[2], kx, ky, kz, x, y, z, d) + B_external(t, omega_ext, amp_ext)

    M = np.cross(m_vec, b)
    M_par = np.dot(M, k) * k
    M_perp = M - M_par

    dW_ndt = (M_perp - I * w_z * np.cross(W_n, k)) / It
    dw_zdt = np.dot(M, k)/I - (fric * w_z)

    return np.concatenate((drdt, dvdt, dkdt, dW_ndt, [dw_zdt], dndt))

#set model params
#-----------------------------------------------------------------------------#
#set base magnet params
R1=0.04 #outer radius
R2=0.02 #inner radius
L=0.012 #height
d = 0.0001

#generate current elements and their position
#-----------------------------------------------------------------------------#
M=3.09620098E+05
global x,y,z,kx,ky,kz
x,y,z,kx,ky,kz=comp.ring(R2,R1,L,M,0,d)
x=np.array(x)
y=np.array(y)
z=np.array(z)
kx=np.array(kx)
ky=np.array(ky)
kz=np.array(kz).astype(np.float64)


z_0=0.058 #initial height
r = [0.001, 0, z_0] #initial position
v = [0, 0, 0] #initial velocitys
ka = [0, 0, 1] #initial rot axis
w_n = [0, 0, 0] #initial trans freq
n = [0, 1, 0]
I=1.633*10**-6 #moment of intertia 
It=0.865*10**-6 #trans moment of intertia
fric = 4.53e-3 #friction coefficient
omega = 190 #spinning freq
u=0.9 #top mass
angle=0 #missaligment angle of axis ang magnetic moment in degrees
m=stcheck.get_mass(z_0, u, kx, ky, kz, x, y, z, d) #set mass
print(m)

#set external field params
amp_ext = 0
omega_ext = omega

stcheck.stability_check(z_0, kx, ky, kz, x, y, z, d) #checking if is top in stable region
stcheck.lower_spin(r, u, I, It, kx, ky, kz, x, y, z,d) #printing minimal allowed frequency 
stcheck.upper_spin(r, u, m, I, kx, ky, kz, x, y, z,d) #printing maximal allowed frequency 

dt=0.001 #dif.eq. time step
T=2000*dt #set time of simulation

f_i=np.concat((r, v, ka, w_n, [omega], n)) #init. state vec
xk = []
yk = []
xt = []
yt = []
zt = []
omegat = []
omegaprec = []
phi =[]

#solver cycle
#-----------------------------------------------------------------------------#
for i in range(int(T/dt)):
   f_i=comp.RK4_step(f_i,Fv,dt,I,It,u,m,9.8,fric,np.deg2rad(angle),i*dt,amp_ext,omega_ext)
   zt.append(f_i[2])
   xk.append(f_i[6])
   yk.append(f_i[7])
   xt.append(f_i[0])
   yt.append(f_i[1])
   omegat.append(f_i[12])
   phi.append(np.atan2(f_i[7],f_i[6]))
   print(i)

#plotting
#-----------------------------------------------------------------------------#
plt.subplot(3 , 2, 1)
plt.plot(np.linspace(0,i*dt,i+1), zt)
plt.subplot(3, 2, 2)
plt.plot(xk,yk)
plt.subplot(3, 2, 3)
plt.plot(xt,yt) 
plt.subplot(3, 2, 4)
plt.plot(np.linspace(0,i*dt,i+1),omegat)  
plt.show()
