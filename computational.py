from numba import jit, njit, float64
import numpy as  np

"""
numeric magnetic field solver using biot-savart law for one point
ox,oy,oz-observation point, 3 numbers
kx,ky,kz-currents vectors, 3 1xM vectors
sx,sy,sz-currents pos vectors, 3 1xM vectors
d-lenght of surface element 
"""

@njit(float64[:](float64, float64, float64,
                 float64[:], float64[:], float64[:],
                 float64[:], float64[:], float64[:],
                 float64))
def Get_B_point(ox, oy, oz, kx, ky, kz, sx, sy, sz, d):

    c = 1e-7
    dS = d * d
    M = kx.shape[0]

    Bx = 0.0
    By = 0.0
    Bz = 0.0

    for i in range(M):
        # Difference vector R
        rx = ox - sx[i]
        ry = oy - sy[i]
        rz = oz - sz[i]

        rmag = (rx*rx + ry*ry + rz*rz)**0.5
        r3 = rmag*rmag*rmag

        # Cross product K × R
        cx = ky[i]*rz - kz[i]*ry
        cy = kz[i]*rx - kx[i]*rz
        cz = kx[i]*ry - ky[i]*rx

        Bx += cx / r3
        By += cy / r3
        Bz += cz / r3

    Bx *= c * dS
    By *= c * dS
    Bz *= c * dS

    return np.array([Bx, By, Bz])

"""
rk4 numeric diff.equation solver step
f_i-input state vector
f-right side of equation
use f in format f(f_i,*args)
dt-time step
*args-other arguments of of f
"""
import numpy as np

def RK4_step(f_i,f,dt,*args):
    f_i=np.array(f_i)
    k1=f(f_i,*args)
    k2=f(f_i+k1*dt/2,*args)
    k3=f(f_i+k2*dt/2,*args)
    k4=f(f_i+k3*dt,*args)
    return f_i+(dt/6)*(k1+2*k2+2*k3+k4)

"""
numeric gradient using central difference
x,y,z-coordinates of difference point
h-differentiation step
f-function to differentiate
use f in format f(x,y,z,*args)
*args-other arguments of of f
"""
import numpy as np
from numba import jit, njit

def gradient_point(x,y,z,h,f,*args):
    g_x=(f(x+h,y,z,*args)-f(x-h,y,z,*args))/(2*h)
    g_y=(f(x,y+h,z,*args)-f(x,y-h,z,*args))/(2*h)
    g_z=(f(x,y,z+h,*args)-f(x,y,z-h,*args))/(2*h)
    return np.array([g_x,g_y,g_z])

def ring(r1,r2,h,M_0,z_0,d):
    pi=np.pi
    x=[]
    y=[]
    z=[]
    k_x=[]
    k_y=[]
    k_z=[]
    s=int(h/d)
    #inner cylinder
    s1=int((2*pi*r1)/d)
    for k in range(s):
         for i in range(s1):
                x.append(r1*np.cos(((2*pi)/s1)*i))
                y.append(r1*np.sin(((2*pi)/s1)*i))
                z.append(k*(h/s)+z_0)
                #magnet proud
                k_x.append(M_0*np.sin(((2*pi)/s1)*i))
                k_y.append(-M_0*np.cos(((2*pi)/s1)*i))
                k_z.append(0)

    #outer cylinder
    s2=int((2*pi*r2)/d)
    for k in range(s):
         for i in range(s2):
                x.append(r2*np.cos(((2*pi)/s2)*i))
                y.append(r2*np.sin(((2*pi)/s2)*i))
                z.append(k*(h/s)+z_0)
                #magnet proud
                k_x.append(-M_0*np.sin(((2*pi)/s2)*i))
                k_y.append(M_0*np.cos(((2*pi)/s2)*i))
                k_z.append(0)

  
    return x,y,z,k_x,k_y,k_z 