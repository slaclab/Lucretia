function D=dmat(L,theta,id);
%
%  Compute dispersion transport matrix for bend or drift
%
%     L = effective length
%     theta = bend angle
%     id = 0: drift
%          1: sbend with e1=e2=theta/2
%          2: sbend with e1=0,e2=theta
%          3: sbend with e1=theta,e2=0

if ((id<0)|(id>3))
   error('  Invalid value for id')
end
if (id==0)
   D=[1 L 0;0 1 0;0 0 1];
   return
end
switch id
   case 1
      beta1=theta/2;
      beta2=theta/2;
   case 2
      beta1=0;
      beta2=theta;
   case 3
      beta1=theta;
      beta2=0;
end
rho=L/theta;
C=cos(theta);
S=sin(theta);
body=[C rho*S rho*(1-C);-S/rho C S;0 0 1];
edge1=[1 0 0;tan(beta1)/rho 1 0;0 0 1];
edge2=[1 0 0;tan(beta2)/rho 1 0;0 0 1];
D=edge2*body*edge1;
