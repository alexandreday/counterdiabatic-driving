function x=conjgradgauge()

%This function implements conjugate gradient minimization for the
%variational gauge potential. 


%Construct function that gives back [[A,H],H]+[[A,Hb],Hb], where Hb is the
%boundary contributions
function B=grad(A,H,H2,HbL,HbL2,HbR,HbR2)
    B=(A*H2+H2*A-2*H*A*H)+(A*HbL2+HbL2*A-2*HbL*A*HbL)+(A*HbR2+HbR2*A-2*HbR*A*HbR);
    
end


for j=1:8
N=j;
tic

%Generate spin matrices
[Sxloc,~,Szloc]=GenerateSparse(N);

%Generate Hamiltonian (open boundary!)
H0=sparse(2^N,2^N);Z=H0;X=H0;
for i=1:N
    if i~=N
        H0=H0+Szloc{i+1}*Szloc{i};
    end
    Z=Z+Szloc{i};
    X=X+Sxloc{i};
end

%Hamiltonian and squared Hamiltonian
H=H0+0.908*Z+0.8*X;
H2=H^2;

%Hlam (couples to lambda) and Hb (boundary Hamiltonian)
Hlam=X;b=1i*(Hlam*H-H*Hlam);normnum(j)=trace(b'*b)/2^N;
HbL=Szloc{1}*sqrt(2);HbR=Szloc{end}*sqrt(2);
HbL2=HbL^2;HbR2=HbR^2;

%Initialize conjugate gradient method, A=operator of interst, r= residue,
%p=conjugate to A.
A=b;
r=b-grad(A,H,H2,HbL,HbL2,HbR,HbR2);
p=r; 
err=trace(r'*r)/2^N;
i=1;
while err>1E-4 && i<4^N
    a=trace(r'*r);
    B=grad(p,H,H2,HbL,HbL2,HbR,HbR2);
    alpha=a/trace(p'*B);
    A=A+alpha*p;
    r=r-alpha*B;
    c=trace(r'*r);err=c/2^N;
    beta=c/a;
    p=r+beta*p;
    i=i+1;
end

steps(j)=i-1;
metricav(j)=trace(A'*A)/2^N;
% %Compute exact A for closed system
% [V,D]=eig(full(H));
% E=diag(D);dE=E*ones(1,2^N)-ones(2^N,1)*E';dE=dE+1/eps*eye(2^N);
% Ae=(V'*Hlam*V)./(dE);
% metricex(j)=trace(Ae'*Ae)/2^N;
% time(j)=toc;

end
plot(metricav);

end