function [B,Bpsi]=bmag(b0,a0,b,a)
%
% [B,Bpsi]=bmag(b0,a0,b,a);
%
% Compute BMAG and its phase from Twiss parameters
%
% INPUTs:
%
%   b0 = matched beta
%   a0 = matched alpha
%   b  = mismatched beta
%   a  = mismatched alpha
%
% OUTPUTs:
%
%   B    = mismatch amplitude
%   Bpsi = mismatch phase (deg)

g0=(1+a0.^2)./b0;
g=(1+a.^2)./b;
B=(b0.*g-2..*a0.*a+g0.*b)/2;
if nargout>1
  Bcos=((b./b0)-B)/sqrt(B.^2-1);
  Bsin=(a-(b./b0).*a0)/sqrt(B.^2-1);
  Bpsi=atan2d(Bsin,Bcos)./2;
end
