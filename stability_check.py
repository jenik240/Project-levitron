from numba import jit, njit, vectorize
import numpy as np
import computational as comp

def stability_check(z_0,kx,ky,kz,x,y,z,d):
    h=0.0001
    s=(comp.Get_B_point(0,0,z_0+h,kx,ky,kz,x,y,z,d)-comp.Get_B_point(0,0,z_0-h,kx,ky,kz,x,y,z,d))[2]/(2*h)
    k=0.5*((comp.Get_B_point(0,0,z_0+h,kx,ky,kz,x,y,z,d)-2*comp.Get_B_point(0,0,z_0,kx,ky,kz,x,y,z,d)+comp.Get_B_point(0,0,z_0-h,kx,ky,kz,x,y,z,d))/h**2)[2]
    B_0=comp.Get_B_point(0,0,z_0,kx,ky,kz,x,y,z,d)[2]
    check=((s/2)**2/(B_0*k))-1
    if check>0:print(f"{check}>0 -> system can be stable here")
    else:print(f"{check}<=0 -> system can't be stable here")

def get_mass(z_0,u,kx,ky,kz,x,y,z,d):
    h=0.0001
    s=(comp.Get_B_point(0,0,z_0+h,kx,ky,kz,x,y,z,d)-comp.Get_B_point(0,0,z_0-h,kx,ky,kz,x,y,z,d))[2]/(2*h)
    m=-(s*u)/9.81
    return m

def upper_spin(r_0,u,m,I,kx,ky,kz,x,y,z,d):
    r_eff_q=I/m
    B_0=np.linalg.norm(comp.Get_B_point(r_0[0],r_0[1],r_0[2],kx,ky,kz,x,y,z,d))
    omega=0.77*(1/(r_eff_q*9.8))*((u*B_0)/m)**(3/2)
    print(f"upper spin limit is {omega}")

def lower_spin(r_0,u,I,It,kx,ky,kz,x,y,z,d):
    B_0 = np.linalg.norm(comp.Get_B_point(r_0[0],r_0[1],r_0[2],kx,ky,kz,x,y,z,d))
    omega = 2 * np.sqrt(u * B_0 * (It/I**2))
    print(f"lower spin limit is {omega}")
