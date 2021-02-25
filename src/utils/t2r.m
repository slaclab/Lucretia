function R=t2r(E1,P1,E2,P2,mu)
%
% Compute pseudo-4x4 R-matrix (diagonal 2x2's) between two points from
% twiss parameters.
%
% (NOTE: Pn=[psix,betax,alphax,etax,etapx,psiy,betay,alphay,etay,etapy])
%
% INPUTs:
%
%   E1 = energy at point 1 (GeV)
%   P1 = twiss at point 1
%   E2 = energy at point 2 (GeV)
%   P2 = twiss at point 2
%   mu = (optional) if provided and nonzero, convert phases in tune units
%        to radians
%
% OUTPUT:
%
%   R=[R11 R12  0   0
%      R21 R22  0   0  
%       0   0  R33 R34
%       0   0  R43 R44]

if (nargin<5)
  mu=0;
end
if (mu)
  conv=2*pi;
else
  conv=1;
end

R=eye(4);

psi1=conv*P1(1);beta1=P1(2);alpha1=P1(3);
psi2=conv*P2(1);beta2=P2(2);alpha2=P2(3);
dpsi=psi2-psi1;
S=sin(dpsi);
C=cos(dpsi);
R(1,1)=sqrt(beta2/beta1)*(C+alpha1*S);
R(1,2)=sqrt(beta1*beta2)*S;
R(2,1)=-((1+alpha1*alpha2)*S+(alpha2-alpha1)*C)/sqrt(beta1*beta2);
R(2,2)=sqrt(beta1/beta2)*(C-alpha2*S);

psi1=conv*P1(6);beta1=P1(7);alpha1=P1(8);
psi2=conv*P2(6);beta2=P2(7);alpha2=P2(8);
dpsi=psi2-psi1;
S=sin(dpsi);
C=cos(dpsi);
R(3,3)=sqrt(beta2/beta1)*(C+alpha1*S);
R(3,4)=sqrt(beta1*beta2)*S;
R(4,3)=-((1+alpha1*alpha2)*S+(alpha2-alpha1)*C)/sqrt(beta1*beta2);
R(4,4)=sqrt(beta1/beta2)*(C-alpha2*S);

R=sqrt(E1/E2)*R;
