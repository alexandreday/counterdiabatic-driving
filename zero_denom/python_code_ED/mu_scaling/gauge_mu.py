from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math function
import time

t_start=time.time()
def gauge_potent_mu(Ham,L, mu):
    E,V= Ham.eigh()
    op_lamb= del_lambda_Ham(L)
    wij = (np.outer(E,np.ones(2**L))-np.outer(np.ones(2**L),E))
    num_lamb = np.dot(V,np.dot(op_lamb.toarray(),np.conj(V)))
    wij=wij+np.identity(2**L)
    A_lamb = -1j*num_lamb*wij/(wij**2+ mu**2)
    np.fill_diagonal(A_lamb, 0.0)
    return A_lamb


def Ham_nonint(L):
    basis = spin_basis_1d(L)
    hz=(np.sqrt(5)+1)/4 #parameters used by Kim and Huse
    hx=(np.sqrt(5)+5)/8
    J=1.0
    hz_arr = [[hz,i] for i in range(L)] # OBC
    hx_arr = [[hx,i] for i in range(L)] # OBC
    J_arr = [[J,i,(i+1)%L] for i in range(L)] # PBC
    
    # static and dynamic lists
    static = [["xx",J_arr],["z",hz_arr], ["x",hx_arr] ]
    dynamic =[]
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return H

def Ham_int(L):
    basis = spin_basis_1d(L)
    hz=(np.sqrt(5)+1)/4 #parameters used by Kim and Huse
    hx=0.0#(np.sqrt(5)+5)/8
    J=1.0
    hz_arr = [[hz,i] for i in range(L)] # OBC
    hx_arr = [[hx,i] for i in range(L)] # OBC
    J_arr = [[J,i,(i+1)%L] for i in range(L)] # PBC
    
    # static and dynamic lists
    static = [["xx",J_arr],["z",hz_arr], ["x",hx_arr] ]
    dynamic =[]
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return H

def del_lambda_Ham(L):
    basis = spin_basis_1d(L)
    hx_lamb=1.0
    hx_lamb_arr = [[hx_lamb,i] for i in range(L)] # OBC
    static_lamb = [["x",hx_lamb_arr]]
    dynamic_lamb =[]
    op_lamb=hamiltonian(static_lamb,dynamic_lamb,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return op_lamb



def norm(A_lamb):    
    return np.linalg.norm(A_lamb, 'fro')


L=12
muTot=20
mu_arr=np.logspace(-4,1.0,muTot)
norm_arr_int=np.zeros(muTot)
H=Ham_int(L)

for i in range(muTot):
    mu=mu_arr[i]
    A_lamb=gauge_potent_mu(H,L,mu)
    norm_arr_int[i]=norm(A_lamb)

H=Ham_nonint(L)
norm_arr_nonint=np.zeros(muTot)
for i in range(muTot):
    mu=mu_arr[i]
    A_lamb=gauge_potent_mu(H,L,mu)
    norm_arr_nonint[i]=norm(A_lamb)
    
t_end=time.time()

t_code=(t_end-t_start)/60


f=open('Int_L12.dat','w')
f.write(" L=%d, code time=%f (in min) \n"  %(L,t_code))
f.write('"mu" \t \t "||A||" \n')

np.savetxt(f, np.transpose([mu_arr,norm_arr_int]) , fmt='%.12f', delimiter='\t')
f.close()

f=open('Non_int_L12.dat','w')
f.write(" L=%d, code time=%f (in min) \n"  %(L,t_code))
f.write('"mu" \t \t "||A||" \n')


np.savetxt(f, np.transpose([mu_arr,norm_arr_nonint]) , fmt='%.12f', delimiter='\t')
f.close()






