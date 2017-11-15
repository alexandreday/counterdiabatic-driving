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
mu=1e-60
hz=5.0
L_arr=range(2,4)
norm_arr_nonint=np.zeros(len(L_arr))
norm_arr_int=np.zeros(len(L_arr))
wij_min_nonint= np.zeros(len(L_arr)) 
wij_max_nonint= np.zeros(len(L_arr))  
wij_min_int= np.zeros(len(L_arr)) 
wij_max_int= np.zeros(len(L_arr))  
###
t_start_nonint=time.time()
###nonint
###finding minimum and maximum wij
###running the loop    
for i in range(0):#len(L_arr)):
	L=L_arr[i]
	H=Ham_nonint(L)
	wij, num_lamb_mat=output_gauge_potent(H,L)
	index_lower = np.tril_indices(2**L,-1)
	wij_arr=wij[index_lower]
	wij_min_nonint[i]= min(wij_arr)
	wij_max_nonint[i]=max(wij_arr)
	A_lamb=gauge_potent_mu(wij, num_lamb_mat,mu)
	norm_arr_nonint[i]=norm(A_lamb)**2/2**L
    


t_end_nonint=time.time()
t_nonint_code=(t_end_nonint-t_start_nonint)/60  

#######
t_start_int=time.time()
###int ham
for i in range(len(L_arr)):
	L=L_arr[i]
	H=Ham_int_antiferro(L,hz)
	wij, num_lamb_mat=output_gauge_potent(H,L)
	index_lower = np.tril_indices(2**L,-1)
	wij_arr=wij[index_lower]
	wij_min_int[i]= min(wij_arr)
	wij_max_int[i]=max(wij_arr)
	A_lamb=gauge_potent_mu(wij, num_lamb_mat,mu)
	norm_arr_int[i]=norm(A_lamb)**2/2**L

    
t_end_int=time.time()

t_int_code=(t_end_int-t_start_int)/60    



#f=open('v2_mu%s_nonint_L_scaling.dat' %mu,'w')
#f.write(" mu=%e, code time=%f (in min) \n"  %(mu,t_nonint_code))
#f.write('"mu" \t \t "||A||^2/2^L \t \t wij_min_nonint \t \t wij_max_nonint" \n')


#np.savetxt(f, np.transpose([L_arr,norm_arr_nonint,wij_min_nonint,wij_max_nonint ]) , fmt='%.20e', delimiter='\t')
#f.close()


f=open('Testing_Analy_v2_mu%s_int_L_scaling.dat' %mu, 'w')
f.write("mu=%e, code time=%f (in min) \n"  %(mu,t_int_code))
f.write('"mu" \t \t "||A||^2/2^L \t \t wij_min_int \t \t wij_max_int" \n')

np.savetxt(f, np.transpose([ L_arr,norm_arr_int, wij_min_int,wij_max_int]) , fmt='%.20e', delimiter='\t')
f.close()


