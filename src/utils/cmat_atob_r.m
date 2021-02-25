function Cab=cmat_atob_r(ida,idb,rmat)
%
% Cab=cmat_atob_r(ida,idb,rmat);
%
% Compute 4x4 closed orbit matrix between two locations
%
% INPUTS:
%
%  ida  = pointer to first location ("a")
%  idb  = pointer to second location ("b")
%  rmat = array of (6x6) R-matrices
%
% OUTPUTS:
%
%  Cab = closed orbit matrix between "a" and "b" (4x4)

Ra=rmat(6*ida-5:6*ida,:);Ra4=Ra(1:4,1:4);
Rb=rmat(6*idb-5:6*idb,:);Rb4=Rb(1:4,1:4);
Rr=rmat(end-5:end,:);Rr4=Rr(1:4,1:4);
I4=eye(4);

if (ida>idb)

% device "a" is downstream of device "b"

  Rm4=Rr4;
else

% device "a" is upstream of device "b", or they're at the same location

  Rm4=I4;
end

% compute Cab

Cab=Rb4*Rm4*inv(Ra4*(I4-Rr4));
