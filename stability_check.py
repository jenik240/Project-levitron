from numba import jit, njit, vectorize
import numpy as np
import computational as comp

"""
Evaluate the analytical stability condition of the levitating system.

z_0                 - equilibrium height
kx,ky,kz            - current element directions
x,y,z               - coordinates of current elements
d                   - discretization step

The function evaluates the quantities S, K, and B_0
using numerical differentiation and checks the condition

((S/2)^2)/(B_0*K) - 1 > 0

which determines whether stable levitation is theoretically possible.
"""
def stability_check(z_0,kx,ky,kz,x,y,z,d):
    h=0.0001
    s=(comp.Get_B_point(0,0,z_0+h,kx,ky,kz,x,y,z,d)-comp.Get_B_point(0,0,z_0-h,kx,ky,kz,x,y,z,d))[2]/(2*h)
    k=0.5*((comp.Get_B_point(0,0,z_0+h,kx,ky,kz,x,y,z,d)-2*comp.Get_B_point(0,0,z_0,kx,ky,kz,x,y,z,d)+comp.Get_B_point(0,0,z_0-h,kx,ky,kz,x,y,z,d))/h**2)[2]
    B_0=comp.Get_B_point(0,0,z_0,kx,ky,kz,x,y,z,d)[2]
    check=((s/2)**2/(B_0*k))-1
    if check>0:print(f"{check}>0 -> system can be stable here")
    else:print(f"{check}<=0 -> system can't be stable here")

"""
Compute the mass required for equilibrium at height z_0.

z_0                 - equilibrium height
u                   - magnetic dipole moment magnitude
kx,ky,kz            - current element directions
x,y,z               - coordinates of current elements
d                   - discretization step

The mass is chosen so that the net vertical force
acting on the top at z_0 is zero.
"""
def get_mass(z_0,u,kx,ky,kz,x,y,z,d):
    h=0.0001
    s=(comp.Get_B_point(0,0,z_0+h,kx,ky,kz,x,y,z,d)-comp.Get_B_point(0,0,z_0-h,kx,ky,kz,x,y,z,d))[2]/(2*h)
    m=-(s*u)/9.81
    return m

"""
Estimate the upper spinning frequency stability limit.

r_0                 - equilibrium position vector
u                   - magnetic dipole moment magnitude
m                   - top mass
I                   - moment of inertia
kx,ky,kz            - current element directions
x,y,z               - coordinates of current elements
d                   - discretization step

The upper stability limit is evaluated using
the analytical approximation derived for the Levitron system.
"""
def upper_spin(r_0,u,m,I,kx,ky,kz,x,y,z,d):
    r_eff_q=I/m
    B_0=np.linalg.norm(comp.Get_B_point(r_0[0],r_0[1],r_0[2],kx,ky,kz,x,y,z,d))
    omega=0.77*(1/(r_eff_q*9.8))*((u*B_0)/m)**(3/2)
    print(f"upper spin limit is {omega}")

"""
Estimate the lower spinning frequency stability limit.

r_0                 - equilibrium position vector
u                   - magnetic dipole moment magnitude
I                   - moment of inertia
It                  - transverse moment of inertia
kx,ky,kz            - current element directions
x,y,z               - coordinates of current elements
d                   - discretization step

The lower stability limit is evaluated from
the analytical approximation of the magnetic torque balance.
"""
def lower_spin(r_0,u,I,It,kx,ky,kz,x,y,z,d):
    B_0 = np.linalg.norm(comp.Get_B_point(r_0[0],r_0[1],r_0[2],kx,ky,kz,x,y,z,d))
    omega = 2 * np.sqrt(u * B_0 * (It/I**2))
    print(f"lower spin limit is {omega}")
