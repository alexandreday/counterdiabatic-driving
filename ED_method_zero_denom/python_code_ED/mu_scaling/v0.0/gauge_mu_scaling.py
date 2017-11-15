from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import time

t_start=time.time()

def Ham_nonint(L):
    basis = spin_basis_1d(L)
    hz=(np.sqrt(5)+1)/4 #parameters used by Kim and Huse
    hx=(np.sqrt(5)+5)/8
    J=1.0
    hz_arr = [[hz,i] for i in range(L)] 
    hx_arr = [[hx,i] for i in range(L)] 
    J_arr = [[1,i,(i+1)] for i in range(L-1)] # OBC [[J,i,(i+1)%L] for i in range(L)] # PBC
    
    # static and dynamic lists
    static = [["xx",J_arr],["z",hz_arr], ["x",hx_arr] ]
    dynamic =[]
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return H

def Ham_int(L):
    basis = spin_basis_1d(L)
    hz=-5.0#(np.sqrt(5)+1)/4 #parameters used by Kim and Huse
    J=-1.0
    hz_arr = [[hz,i] for i in range(L)] # OBC
    J_arr =  [[1,i,(i+1)] for i in range(L-1)] # OBC[[J,i,(i+1)%L] for i in range(L)] # PBC
    
    # static and dynamic lists
    static = [["xx",J_arr],["z",hz_arr]]
    dynamic =[]
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return H

def del_lambda_Ham(L):
    basis = spin_basis_1d(L)  
    hx_lamb=-1.0
    hx_lamb_arr=np.zeros(L)
    hx_lamb_arr[0]=hx_lamb
    hx_lamb_arr = [[hx_lamb_arr[i],i] for i in range(L)] # OBC
    static_lamb = [["z",hx_lamb_arr]]
    dynamic_lamb =[]
    op_lamb=hamiltonian(static_lamb,dynamic_lamb,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return op_lamb

def norm(A_lamb):    
    return np.linalg.norm(A_lamb, 'fro')
    
def gauge_potent_mu(Ham,L, mu):
    E,V= Ham.eigh()
    op_lamb= del_lambda_Ham(L)
    wij = np.outer(E,np.ones(2**L))-np.outer(np.ones(2**L),E)
    num_lamb = np.dot(V,np.dot(op_lamb.toarray(),np.conj(V)))
    A_lamb = -1j*num_lamb*wij/(wij**2+ mu**2)
    return A_lamb    
    


L=8
mu_L8=np.logspace(-5,2.0,20)
Ntot=len(mu_L8)
H=Ham_int(L)

norm_arr_int8=np.zeros(Ntot)

for i in range(Ntot):
    A_lamb=gauge_potent_mu(H,L,mu_L8[i])
    norm_arr_int8[i]=norm(A_lamb)
    
    
H=Ham_nonint(L)
norm_arr_nonintL8=np.zeros(Ntot)
for i in range(Ntot):
	A_lamb=gauge_potent_mu(H,L,mu_L8[i])
	norm_arr_nonintL8[i]=norm(A_lamb)
        
    
t_end=time.time()

t_code=(t_end-t_start)/60    


f=open('Int_mu_scaling.dat','w')
f.write(" L=%d, code time=%f (in min) \n"  %(L,t_code))
f.write('"mu^2" \t \t "||A||^2" \n')

np.savetxt(f, np.transpose([ mu_L8**2,norm_arr_int8**2]) , fmt='%.12f', delimiter='\t')
f.close()

f=open('Non_int_mu_scaling.dat','w')
f.write(" L=%d, code time=%f (in min) \n"  %(L,t_code))
f.write('"mu^2" \t \t "||A||^2" \n')


np.savetxt(f, np.transpose([mu_L8**2,norm_arr_nonintL8**2]) , fmt='%.12f', delimiter='\t')
f.close()



