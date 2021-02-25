function coef=bumps(plane,flavor,id0,idc,P,E)
%
% coef=bumps(plane,flavor,id0,idc,P,E);
%
% Compute coefficients for an orbit bump.
%
% Inputs:
%
%  plane  = x or y plane flag (0=x,1=y)
%  flavor = position or angle flag (0=pos,1=ang)
%  id0    = pointer to target location in P and E arrays
%  idc    = pointers to correctors in P and E  arrays (3 or 4 values)
%  P      = twiss parameter array (i.e. from COMF2MAT or GET_NEW_MODEL)
%  E      = energy array (i.e. from COMF2MAT or GET_NEW_MODEL)
%
% Outputs:
%
%  coef   = corrector coefficients (mrad/mm_of_bump or mrad/mrad_of_bump)
%
% Assumptions:
%
%  - 3 corrector: --c1--c2--0--c3-- or --c1--0--c2--c3--
%  - 4 corrector: --c1--c2--0--c3--c4--
%  - no x-y coupling
%

% check x/y plane arg ... set up R-matrix offset

if (plane==0)
  psi0=P(id0,1);
  psic=P(idc,1);
  ioff=0;
elseif (plane==1)
  psi0=P(id0,6);
  psic=P(idc,6);
  ioff=2;
else
  error('plane should be 0 or 1')
end

% check angle/position arg

if (flavor==0)
  pos=1;
elseif (flavor==1)
  pos=0;
else
  error('flavor should be 0 or 1')
end

% check relative locations of correctors and target ... determine bump type

nc=length(idc);
if (nc==3)
  if ((psi0<psic(1))|(psi0>psic(3)))
    error('target must be between first and last corrector')
  elseif (psi0>psic(2))
    bmp=1;  % target between 2nd and 3rd correctors
  else
    bmp=2;  % target between 1st and 2nd correctors
  end
elseif (nc==4)
  if ((psi0<psic(2))|(psi0>psic(3)))
    error('target must be between second and third corrector')
  else
    bmp=3;
  end
else
  error('3 or 4 correctors only')
end

% download twiss and energy values from input arrays

P0=P(id0,:);E0=E(id0);
P1=P(idc(1),:);E1=E(idc(1));
P2=P(idc(2),:);E2=E(idc(2));
P3=P(idc(3),:);E3=E(idc(3));
if (nc==4)
  P4=P(idc(4),:);E4=E(idc(4));
end

% set up Ax=b system

if (bmp~=3)

% 3-corrector bumps

  Ra=t2r(E1,P1,E0,P0);
  if (bmp==1)
    Rb=t2r(E2,P2,E0,P0);  % 2nd corrector upstream of target
  else
    Rb=zeros(4,4);        % 2nd corrector downstream of target
  end
  Rc=t2r(E1,P1,E3,P3);
  Rd=t2r(E2,P2,E3,P3);
  if (pos)
    A=[Ra(1+ioff,2+ioff),Rb(1+ioff,2+ioff),0; ...
       Rc(1+ioff,2+ioff),Rd(1+ioff,2+ioff),0; ...
       Rc(2+ioff,2+ioff),Rd(2+ioff,2+ioff),1];
  else
    A=[Ra(2+ioff,2+ioff),Rb(2+ioff,2+ioff),0; ...
       Rc(1+ioff,2+ioff),Rd(1+ioff,2+ioff),0; ...
       Rc(2+ioff,2+ioff),Rd(2+ioff,2+ioff),1];
  end
  b=[1;0;0];
else

% 4-corrector bumps

  Ra=t2r(E1,P1,E0,P0);
  Rb=t2r(E2,P2,E0,P0);
  Rc=t2r(E1,P1,E4,P4);
  Rd=t2r(E2,P2,E4,P4);
  Re=t2r(E3,P3,E4,P4);
  A=[Ra(1+ioff,2+ioff),Rb(1+ioff,2+ioff),                0,0; ...
     Ra(2+ioff,2+ioff),Rb(2+ioff,2+ioff),                0,0; ...
     Rc(1+ioff,2+ioff),Rd(1+ioff,2+ioff),Re(1+ioff,2+ioff),0; ...
     Rc(2+ioff,2+ioff),Rd(2+ioff,2+ioff),Re(2+ioff,2+ioff),1];
  if (pos)
    b=[1;0;0;0];
  else
    b=[0;1;0;0];
  end
end

% solve Ax=b for x

coef=inv(A)*b;
