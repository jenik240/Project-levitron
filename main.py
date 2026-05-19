from numba import jit, njit, vectorize
from numba import float64
import numpy as np
import matplotlib.pyplot as plt
import computational as comp
import stability_check as stcheck

#compute force on top
#-----------------------------------------------------------------------------#
def B_external(t, omega, amp):
    return amp*np.cos(omega*t)*np.array([0,1,0])

def g(r,u,m,g):
    u=np.array(u)
    mag_force=comp.gradient_point(r[0],r[1],r[2],0.0001,F,u)
    return mag_force/m-g*np.array([0,0,1])

def F(rx,ry,rz,u):
    b=comp.Get_B_point(rx,ry,rz,kx,ky,kz,x,y,z,0.0001) 
    return np.dot(u,b)

def Fv(f_i, I, It, m_0, m, ga, fric, alpha, t):
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

    # magnetický moment pevný v tělese
    m_vec = -m_0 * (np.cos(alpha) * k + np.sin(alpha) * n)

    drdt = v
    dvdt = g(r, m_vec, m, ga)

    dkdt = np.cross(W_n, k)
    dndt = np.cross(omega_vec, n)

    b = comp.Get_B_point(r[0], r[1], r[2], kx, ky, kz, x, y, z, 0.0001) + B_external(t, 200, 0)

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

#generate current elements and their position
#-----------------------------------------------------------------------------#
M=3.09620098E+05
global x,y,z,kx,ky,kz
x,y,z,kx,ky,kz=comp.ring(R2,R1,L,M,0,0.0001)
x=np.array(x)
y=np.array(y)
z=np.array(z)
kx=np.array(kx)
ky=np.array(ky)
kz=np.array(kz).astype(np.float64)


z_0=0.056 #initial height
r = [0.0015, 0, z_0] #initial position
v = [0, 0, 0] #initial velocitys
ka = [0, 0, 1] #initial rot axis
w_n = [0, 0, 0] #initial trans freq
n = [0, 1, 0]
I=1.633*10**-6 #moment of intertia 
It=0.865*10**-6 #trans moment of intertia
fric = 0 #friction coefficient
omega = 190 #spinning freq
u=0.9 #top mass
angle=0 #missaligment angle of axis ang magnetic moment in degrees
m=stcheck.get_mass(z_0, u, kx, ky, kz, x, y, z) #set mass
print(m)

stcheck.stability_check(z_0, kx, ky, kz, x, y, z) #checking if is top in stable region
stcheck.lower_spin(r, u, I, It, kx, ky, kz, x, y, z) #printing minimal allowed frequency 
stcheck.upper_spin(r, u, m, I, kx, ky, kz, x, y, z) #printing maximal allowed frequency 

dt=0.001 #dif.eq. time step
T=500*dt #set time of simulation

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
   f_i=comp.RK4_step(f_i,Fv,dt,I,It,u,m,9.8,fric,np.deg2rad(angle),i*dt)
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
#np.savetxt("data_nostab.txt",np.column_stack((np.linspace(0,i*dt,i+1),zt)))
plt.subplot(3 , 2, 1)
plt.plot(np.linspace(0,i*dt,i+1), zt)
plt.subplot(3, 2, 2)
plt.plot(xk,yk)
plt.subplot(3, 2, 3)
plt.plot(xt,yt) 
plt.subplot(3, 2, 4)
plt.plot(np.linspace(0,i*dt,i+1),omegat)  
plt.show()
