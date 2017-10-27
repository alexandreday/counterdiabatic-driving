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
    J_arr =[[J,i,(i+1)] for i in range(L-1)] # OBC  [[J,i,(i+1)%L] for i in range(L)] # PBC
    
    # static and dynamic lists
    static = [["zz",J_arr],["z",hz_arr], ["x",hx_arr] ]
    dynamic =[]
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return H

def Ham_int_antiferro(L,hz):
    basis = spin_basis_1d(L)
    J=1.0
    hz_arr = [[hz,i] for i in range(L)] # OBC
    J_arr =[[J,i,(i+1)] for i in range(L-1)] # OBC# [[J,i,(i+1)%L] for i in range(L)] # PBC
    # static and dynamic lists
    static = [["zz",J_arr],["x",hz_arr]]
    dynamic =[]
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return H


def del_lambda_Ham(L):
    basis = spin_basis_1d(L)  
    hx_lamb=1.0
    hx_lamb_arr = [[hx_lamb,i] for i in range(L)] 
    static_lamb = [["x",hx_lamb_arr]]
    dynamic_lamb =[]
    op_lamb=hamiltonian(static_lamb,dynamic_lamb,basis=basis,dtype=np.complex_,check_symm=False,check_herm=False)
    return op_lamb

def norm(A_lamb):    
    return np.linalg.norm(A_lamb, 'fro')

def output_gauge_potent(Ham,L):
	E,V= Ham.eigh()
	V_mat=np.matrix(V)
    	V_mat_H=V_mat.H
    	op_lamb_mat = np.matrix(del_lambda_Ham(L).toarray())
    	num_lamb_mat =  (V_mat_H)*(op_lamb_mat*V_mat) #matrix multiplication
    	wij = np.outer(E,np.ones(2**L))-np.outer(np.ones(2**L),E)
	return wij, num_lamb_mat
  
def gauge_potent_mu(wij,num_lamb_mat,mu):
    A_lamb = -1j*np.multiply(wij,num_lamb_mat)/(wij**2+ mu**2)#element-wise multiplication
    return A_lamb

###parameters
muTot=40
mu_arr=np.logspace(-6,1.0,muTot)
L=10
norm_arr_nonint=np.zeros(muTot)
norm_arr_int=np.zeros(muTot)
hz=5.00#for int Ham

###
t_start_nonint=time.time()
###nonint
H=Ham_nonint(L)
wij, num_lamb_mat=output_gauge_potent(H,L)
###finding minimum and maximum wij
index_lower = np.tril_indices(2**L,-1)
wij_arr=wij[index_lower]
wij_min_nonint= min(wij_arr)
wij_max_nonint=max(wij_arr)
print wij_min_nonint,wij_max_nonint
###running the loop    
for i in range(muTot):
    mu=mu_arr[i]
    A_lamb=gauge_potent_mu(wij, num_lamb_mat,mu)
    norm_arr_nonint[i]=norm(A_lamb)


t_end_nonint=time.time()
t_nonint_code=(t_end_nonint-t_start_nonint)/60  

#######
t_start_int=time.time()
###int ham
H=Ham_int_antiferro(L,hz)
wij, num_lamb_mat=output_gauge_potent(H,L)
###finding minimum and maximum wij
index_lower = np.tril_indices(2**L,-1)
wij_arr=wij[index_lower]
wij_min_int= min(wij_arr)
wij_max_int=max(wij_arr)
for i in range(muTot):
    mu=mu_arr[i]
    A_lamb=gauge_potent_mu(wij, num_lamb_mat,mu)
    norm_arr_int[i]=norm(A_lamb)
    
    
t_end_int=time.time()

t_int_code=(t_end_int-t_start_int)/60    



f=open('Test_v2.0_L%s_nonint_mu_scaling.dat' %L,'w')
f.write(" L=%d, code time=%f (in min) \n"  %(L,t_nonint_code))
f.write("wij_min=%.20e, wij_max=%.20e \n"  %(wij_min_nonint,wij_max_nonint))
f.write('"mu" \t \t "||A||^2" \n')


np.savetxt(f, np.transpose([mu_arr,norm_arr_nonint**2]) , fmt='%.20e', delimiter='\t')
f.close()


f=open('Test_v2.0_L%s_int_mu_scaling.dat' %L, 'w')
f.write("L=%d, code time=%f (in min) \n"  %(L,t_int_code))
f.write("wij_min=%.20e, wij_max=%.20e \n"  %(wij_min_int,wij_max_int))
f.write('"mu" \t \t "||A||^2" \n')

np.savetxt(f, np.transpose([ mu_arr,norm_arr_int**2]) , fmt='%.20e', delimiter='\t')
f.close()


