function [p,dp]=waist_params(sig11,sig12,sig22,C)

% [p,dp]=waist_params(sig11,sig12,sig22,C);
%
% Returns distance to nearest waist and beam parameters at the waist (beta and
% size), and errors on these, given fitted sigma11(33), sigma12(34), 
% sigma22(44) and the 3X3 covariance matrix of this fit.
%
% INPUTS:
%
%   sig11 : the fitted 1,1 (3,3) sigma matrix element (in m^2-rad)
%   sig12 : the fitted 1,2 (3,4) sigma matrix element (in m-rad)
%   sig22 : the fitted 2,2 (4,4) sigma matrix element (in rad^2)
%   C     : the 3X3 covariance matrix of the above fitted sig11,12,22
%           (33,34,44) (in squared units of the 3 above sigij's)
%
% OUTPUTS:
%
%   p  : p(1) = distance to nearest waist (m) ... positive = downstream
%        p(2) = beta function at the waist (m)
%        p(3) = beam size at the waist (m)
%   dp : propagated errors on p(1),p(2),p(3)

%===============================================================================

e2=sig11*sig22-sig12^2;
if (e2<0)
  error('waist_params: Negative emittance')
end
e=sqrt(e2);

Lw=-sig12/sig22;
grad=[0;-1/sig22;sig12/sig22^2];
dLw=sqrt(grad'*C*grad);

bw=e/sig22;
grad=[1/(2*e);-sig12/(e*sig22);sig11/(2*e*sig22)-e/(sig22^2)];
dbw=sqrt(grad'*C*grad);

sigw=sqrt(e*bw);
grad=[sqrt(sig22)/(2*e);-sig12/(e*sqrt(sig22)); ...
  sig11/(2*e*sqrt(sig22))-e/(2*sig22*sqrt(sig22))];
dsigw=sqrt(grad'*C*grad);

p=[Lw,bw,sigw];
dp=[dLw,dbw,dsigw];

end
