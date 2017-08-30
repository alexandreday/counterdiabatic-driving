from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import sympy as sp
import time



x,tau_0 = sp.symbols('x tau')
def lambda_fn_sym(t,tau):
    pi=np.pi
    #tau=1.0
    lambda_0=0.0
    lambda_f=-10.0
    return lambda_0 + (lambda_f-lambda_0)*sp.sin(pi/2*(sp.sin(t*pi/2.0/tau)**2))**2


def dot_lambda_fn_sym(t,tau):
    return sp.diff(lambda_fn_sym(t, tau),t)


np_lambda = sp.lambdify((x,tau_0),lambda_fn_sym(x,tau_0), modules=['numpy'])
np_lambda_dot = sp.lambdify((x,tau_0), dot_lambda_fn_sym(x,tau_0), modules=['numpy'])


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
        #print "tau:", tau
        #tau=1
        return t/tau

    def dries_ramp(t):
        global tau
        pi=np.pi
        lambda_0=0.0
        lambda_f=-10.0
        return lambda_0 + (lambda_f-lambda_0)*np.sin(pi/2*(np.sin(t*pi/2.0/tau)**2))**2

    
    def alpha_0(t_var):
        return 1.0/(6+ (t_var+0.8)**2 )

    def CD_ramp(t):
        global tau
        return np_lambda_dot(t,tau)*alpha_0(t)

    ramp_args=[]
    s=np.zeros(L)
    s[0]=1
    J_x_t = [[s[j],j] for j in range(L-1) ]
    J_y_t = [[s[j],j] for j in range(L-1) ]

    dynamic =[["x", J_x_t,dries_ramp,ramp_args], ["y", J_y_t,CD_ramp,ramp_args]]
    # compute the time-dependent Heisenberg Hamiltonian
    H0 = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_)
    #print J_z,'\n', Z, '\n', X
    #print "Ham \n", H0
    return H0

def fidelity(psi_t,psi_gs):
    return np.abs(np.vdot(psi_gs,psi_t))**2

def energy_diff(psi_t,energy_gs, H0_t):
    energy_psi_t= np.vdot(psi_t, H0_t.dot(psi_t))
    return np.real(energy_psi_t-energy_gs)


t_start=time.time()


L=10
tau=0.01
H0=Ham(L)
E0,V0=H0.eigh(time=0)
psi0=V0[:,0]
t_in=np.logspace(-2,1.39794,15,endpoint=True)
fidelity_arr=np.ones(len(t_in))
energy_diff_arr=np.ones(len(t_in))


for i in range(len(t_in)):
	tau=t_in[i]
	psi_t = H0.evolve(psi0,0.0,t_in[i])
	energy,evector=H0.eigh(time=t_in[i])
	psi_gs=evector[:,0]
	energy_gs=energy[0]
	fidelity_arr[i]=fidelity(psi_t,psi_gs)
	energy_diff_arr[i]=energy_diff(psi_t,energy_gs, H0(time=t_in[i]))
	print fidelity_arr[i],energy_diff_arr[i]

t_end=time.time()
t_code=(t_end-t_start)/60.0

f=open('CD_log_dries_ramp.dat','w')
f.write("L= %.12f code ran for= %.2f (in minutes) \n " %(L, t_code))
for i in range(len(t_in)):
	f.write(" %.12f %.12f %.12f  \n"  %(t_in[i],fidelity_arr[i],energy_diff_arr[i]))  


