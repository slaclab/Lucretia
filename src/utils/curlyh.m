function H=curlyh(b,a,n,np,l,theta,k1)

%  H=curlyH(b,a,n,np,l,theta,k1);
%
%  Computes <H> for normal boundary magnets per equation (20) of
%  SLAC-PUB-1193, "Evaluation of Synchrotron Radiation Integrals"
%
%  Inputs:
%
%     b     = beta at entrance of magnet [meter]
%     a     = alpha (= -b'/2) at entrance of magnet
%     n     = eta at entrance of magnet [meter]
%     np    = eta-prime at entrance of magnet
%     l     = length of magnet [meter]
%     theta = bend angle of magnet [radian]
%     k1    = quadrupole strength of magnet [1/meter^2]
%             (k1>0 for focusing, k1<0 for defocusing)

h=theta/l;
Ksq=h^2+k1;
if (Ksq<0)
   K=sqrt(-Ksq);
   Kl=K*l;
   C=cosh(Kl);
   S=sinh(Kl);
   sgn=-1;
else
   K=sqrt(Ksq);
   Kl=K*l;
   C=cos(Kl);
   S=sin(Kl);
   sgn=1;
end
g=(1+a^2)/b;

H=g*n^2+2*a*n*np+b*np^2 ...
 +sgn*2*l*h*(-(g*n+a*np)*(Kl-S)/(K*Kl^2)+(a*n+b*np)*(1-C)/(Kl^2)) ...
 +(l*h)^2*(g*(3*Kl-4*S+S*C)/(2*K^2*Kl^3)-a*(1-C)^2/(K*Kl^3) ...
 +sgn*b*(Kl-C*S)/(2*Kl^3));
