function [X,Y,Z,Yaw,Pitch,Roll]=survey(K,L,P,X0,Y0,Z0,Yaw0,Pitch0,Roll0);

% allocate arrays

X=zeros(size(L));
Y=zeros(size(L));
Z=zeros(size(L));
Yaw=zeros(size(L));
Pitch=zeros(size(L));
Roll=zeros(size(L));

% initialize displacement vector and rotation matrix

V=[X0;Y0;Z0];
theta=Yaw0;
c=cos(theta);
s=sin(theta);
Wtheta=[c,0,s;0,1,0;-s,0,c];
phi=Pitch0;
c=cos(phi);
s=sin(phi);
Wphi=[1,0,0;0,c,s;0,-s,c];
psi=Roll0;
c=cos(psi);
s=sin(psi);
Wpsi=[c,-s,0;s,c,0;0,0,1];
W=Wtheta*Wphi*Wpsi;

% initial coordinates

X(1)=X0;
Y(1)=Y0;
Z(1)=Z0;
Yaw(1)=Yaw0;
Pitch(1)=Pitch0;
Roll(1)=Roll0;

% loop over elements
% (NOTE: assume INITIAL element is first)

for n=2:length(L)

% set up element displacement and rotation

  switch K(n,:)
    case 'SBEN'
      alpha=P(n,1);
      c=cos(alpha);
      s=sin(alpha);
      if (alpha==0)
        R=[0;0;L(n)];
      else
        rho=L(n)/alpha;
        R=[rho*(c-1);0;rho*s];
      end
      S=[c,0,-s;0,1,0;s,0,c];
      tilt=P(n,4);
      c=cos(tilt);
      s=sin(tilt);
      T=[c,-s,0;s,c,0;0,0,1];
      R=T*R;
      S=T*S*inv(T);
    case 'RBEN'
      alpha=P(n,1);
      c=cos(alpha);
      s=sin(alpha);
      if (alpha==0)
        R=[0;0;L(n)];
      else
        rho=L(n)/(2*sin(alpha/2));
        R=[rho*(c-1);0;rho*s];
      end
      S=[c,0,-s;0,1,0;s,0,c];
      tilt=P(n,4);
      c=cos(tilt);
      s=sin(tilt);
      T=[c,-s,0;s,c,0;0,0,1];
      R=T*R;
      S=T*S*inv(T);
    case 'MULT'
      alpha=P(n,1);
      c=cos(alpha);
      s=sin(alpha);
      R=[0;0;0];
      S=[c,0,-s;0,1,0;s,0,c];
      tilt=P(n,4);
      c=cos(tilt);
      s=sin(tilt);
      T=[c,-s,0;s,c,0;0,0,1];
      R=T*R;
      S=T*S*inv(T);
    case 'SROT'
      R=[0;0;0];
      angle=P(n,5);
      c=cos(angle);
      s=sin(angle);
      S=[c,-s,0;s,c,0;0,0,1];
    case 'YROT'
      R=[0;0;0];
      angle=P(n,5);
      c=cos(angle);
      s=sin(angle);
      S=[c,0,-s;0,1,0;s,0,c];
    otherwise
      R=[0;0;L(n)];
      S=eye(3);
  end

% update global displacement and rotation

  V=W*R+V;
  W=W*S;

% compute survey angles from global rotation

  arg=sqrt(W(2,1)^2+W(2,2)^2);
  phi=atan2(W(2,3),arg);
  if (arg>1e-20)
    theta=proxim(atan2(W(1,3),W(3,3)),theta);
    psi=proxim(atan2(W(2,1),W(2,2)),psi);
  else
    psi=proxim(atan2(-W(1,2),W(1,1))-theta,psi);
  end

% survey coordinates at exit of element

  X(n)=V(1);
  Y(n)=V(2);
  Z(n)=V(3);
  Yaw(n)=theta;
  Pitch(n)=phi;
  Roll(n)=psi;
end

function p=proxim(a,b)
twopi=2*pi;
p=a+twopi*round((b-a)/twopi);