function Cab=cmat_atob(twssa,twssb,twssr)
%
% Cab=cmat_atob(twssa,twssb,twssr);
%
% Compute 4x4 "closed orbit" transfer matrix between devices "a" and "b"
% for a storage ring from twiss parameters
%
% INPUTS:
%
%  twssa = twiss parameters for device "a"
%  twssb = twiss parameters for device "b"
%  twssr = twiss parameters for entire ring
%
% OUTPUTS:
%
%  Cab   = "Cij" matrix from device "a" to device "b"

% construct R-matrices

twss0=twssr;
twss0(1)=0;
twss0(6)=0;
Ra=t2r(1,twss0,1,twssa);  % start of ring to device "a"
Rb=t2r(1,twss0,1,twssb);  % start of ring to device "b"
Rr=t2r(1,twss0,1,twssr);  % start of ring to end of ring

% use phases to determine if device "b" is upstream of device "a"

psixa=twssa(1);
psiya=twssa(6);
psixb=twssb(1);
psiyb=twssb(6);
I=eye(4);
if ((psixb<psixa)&(psiyb<psiya))

% device "b" is upstream of device "a"

  Rm=Rr;
elseif ((psixb>=psixa)&(psiyb>=psiya))

% device "b" is downstream of device "a", or they're at the same location

  Rm=I;
else
  error('Inconsistent x/y phase advance between "a" and "b"')
end

% compute Cab

Cab=Rb*Rm*inv(Ra*(I-Rr));
