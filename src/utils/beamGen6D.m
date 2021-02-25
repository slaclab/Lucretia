function X = beamGen6D(n,mu,sigma)
% X = beamGen6D(n,mu,sigma)
% Generate a lucretia beam distribution (Beam.Bunch.x part)
% given 6-D sigma matrix (sigma) and mean vector (mu) containing (n) particles

% check mu vector is column-wise
[r c]=size(mu); if c>r; mu=mu'; end;
if length(mu)~=6; error('mu should be 6d vector'); end;
[r c]=size(sigma);
if r~=6 || c~=6; error('sigma should be 6x6 beam sigma matrix'); end;

% --- Generate Beam
% Calculate eigenvalues and vectors for sigma matrix
[v,d] = eig(sigma) ;
% Generate macroparticles
X=zeros(6,n);
for iMacro=1:n
  X(:,iMacro)=v*(randn(6,1).*sqrt(diag(d)))+mu;
end % for iMacro