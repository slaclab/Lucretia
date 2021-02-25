E=1.98;
Cb=33.35640952;
Brho=Cb*E;
B=21.0;
lamw=1.0;

L=lamw/4;
D=lamw/4;
Dh=lamw/16;

rho=Brho/B;
theta=2*asin(L/2/rho);
l=rho*theta;
thetah=theta/2;
lh=l/2;
d=D/cos(theta/2);
dh=Dh/cos(thetah);

D1=dmat(lh,-thetah,2);  % -half
D2=dmat(dh,0,0);        % half->full  
D3=dmat(l,theta,1);     % +full
D4=dmat(d,0,0);         % full->full
D5=dmat(l,-theta,1);    % -full
D6=dmat(lh,thetah,3);   % +half

R=D6*D2*D5*D4*D3*D4*D5*D4*D3*D2*D1;


