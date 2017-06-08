from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions




def Ham(L):
    basis = spin_basis_1d(L)
    # static 
    J_z = [[1.0,i,(i+1)%L] for i in range(L)] # PBC
    Z = [[2.0,i] for i in range(L)] 
    X = [[0.8,i] for i in range(L)]
    static = [["zz",J_z], ["z",Z], ["x",X]]

    #dynamic var
    def linear_ramp(t):
	global tau
        return t/tau

    def dries_ramp(t):
        global tau
        p=np.pi
        lambda_0=0.1
        lambda_f=-10.0
        return lambda_0 + (lambda_f-lambda_0)*np.sin(p/2*(np.sin(t*p/2.0/tau)**2))**2

    ramp_args=[]
    s=np.zeros(L)
    s[0]=1
    J_x_t = [[s[j],j] for j in range(L-1) ]

    dynamic =[["x", J_x_t,dries_ramp,ramp_args]]
    # compute the time-dependent Heisenberg Hamiltonian
    H0 = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_)
    return H0


L=10
tau=0.01
H0=Ham(L)
E0,V0=H0.eigh(time=0)
psi0=V0[:,0]
t_in=np.linspace(0.01,10,20)
fidelity_arr=np.ones(len(t_in))
energy_diff_arr=np.ones(len(t_in))

def fidelity(t,psi):
    energy,evector=H0.eigh(time=t)
    psi_gs=evector[:,0]
    return np.abs(np.vdot(psi_gs,psi))**2

def energy_diff(t,psi):
    energy_psi= np.vdot(psi, H0(time=t).dot(psi))
    gs_energy=H0.eigvalsh(time=t)
    return energy_psi-gs_energy[0] 


for i in range(len(t_in)):
	tau=t_in[i]
	psi = H0.evolve(psi0,0.0,t_in[i])	
	fidelity_arr[i]=fidelity(t_in[i],psi)
	energy_diff_arr[i]=np.real(energy_diff(t_in[i],psi))

f=open('dries_ramp.dat','w')
for i in range(len(t_in)):
	f.write(" %.12f %.12f %.12f  \n"  %(t_in[i],fidelity_arr[i],energy_diff_arr[i]))  














