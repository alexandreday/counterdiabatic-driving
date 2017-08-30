   # -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:22:35 2017

@author: Jonathan Wurtz
"""

# Import all sundries beforehand

from quspin.operators import hamiltonian
import scipy.sparse
from numpy import kron
def locop(sp,i,N):
  '''
  Local operator
  sp - operator type: {x,y,z,i}
  i  - site index
  N  - Total number of sites
  '''
  if sp=='i':
      return scipy.sparse.identity(2**N,format='csc')
  else:
      return hamiltonian([[sp,[[1,i]]]],[],N = int(N),check_symm=False,check_herm=False).tocsc()

hx = 1.
hz = 1.
J = 0.5
jj = 0
Nsites_ = [9]#range(5,9)
dout = zeros(len(Nsites_))
for Nsites in Nsites_:
    #Nsites = 5
    
    # Create the Hamiltonian and operator
    ham = 1j*zeros([2**Nsites,2**Nsites])
    op = 1j*zeros([2**Nsites,2**Nsites])
    for i in range(Nsites):
        op += locop('x',i,Nsites).toarray()
        
        ham += hz*random.normal()*locop('z',i,Nsites).toarray()
        #ham += hx*locop('x',i,Nsites).toarray()
        ham += dot(locop('x',i,Nsites).toarray(),locop('x',(i+1)%Nsites,Nsites).toarray())
        ham += dot(locop('y',i,Nsites).toarray(),locop('y',(i+1)%Nsites,Nsites).toarray())
        ham += dot(locop('z',i,Nsites).toarray(),locop('z',(i+1)%Nsites,Nsites).toarray())
    #op = locop('x',0,Nsites).toarray()
    
    # Find the Eigensistem
    ham_eig = linalg.eigh(ham)
    
    #print 'Orthogonality Condition:',(sum(abs(einsum('xa,xb',conj(ham_eig[1]),ham_eig[1])))-2**Nsites)/4**Nsites
    # Operator in the Eigenbasis of the Hamiltonian
    op_eigbasis = dot(ham_eig[1].T,dot(op,conj(ham_eig[1])))
    
    # Scan across effective masses mu_

    dout = zeros(100)
    
    mu_ = logspace(-4,1,len(dout))
    wij = (outer(ham_eig[0],ones(2**Nsites))-outer(ones(2**Nsites),ham_eig[0]))
    for ii in range(len(mu_)):
        mu = mu_[ii]
        #mu = 0
        counterD = op_eigbasis*wij/(wij**2 + mu**2)
        counterD[range(2**Nsites),range(2**Nsites)] = 0
        # Somehow sometimes some values are NaN...?
        counterD[isnan(counterD)] = 0
        #print 'Nsites:      ',Nsites
        #print 'Lognorm of A div Nsites:',log(abs(trace(dot(counterD,counterD))))/Nsites
        dout[ii] = abs(trace(dot(counterD,counterD)))
    loglog(mu_,dout,label="Nsites="+str(Nsites))
    jj+= 1
