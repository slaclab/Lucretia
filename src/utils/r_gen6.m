function R = R_gen6(L,angle,k1,roll,prot)

%    R = R_gen6(L, angle, k1, roll, prot);
%
%    Returns a general 6X6 R matrix (x,x',y,y',dp/p)
%
%
%    where:
%
%       L:      is magnetic length (see below for L=0)  [meters]
%       angle:  is bending angle                        [rads]
%       k1:     is quad strenght                        [1/meter^2]
%                 (k1>0 gives x focusing
%                  k1<0   "   y    "    )
%       roll:   (Opt) is roll angle around longitudinal axis  [rads]
%               (roll>0: roll is clockwise about positive Z-axis.
%                        i.e. clockwise next element rotation as beam
%                        leaves the observer)
%            ==>(NOTE: if L=0, then R = rotation through "roll" angle)
%       prot:   (Opt,DEF=0) If "prot" = 1, this is a rectangular bend, else
%               sector bend (or not bend).
%
%    eg.  R_gen(L,0    ,0 )             gives drift matrix
%         R_gen(L,angle,0 )             gives pure sector dipole
%         R_gen(L,angle,0,0,1 )         gives rectangular bend
%         R_gen(L,0    ,k1)             gives pure quadrupole
%         R_gen(L,0    ,k1,pi/4)        gives skew quadrupole
%         R_gen(L,angle,k1)             gives combo function magnet
%         R_gen(0,0,0,roll)             gives rotation matrix
%         R_gen(L,angle,k1,roll)        gives rotated combo func mag

%===============================================================================

if nargin < 4      
  roll = 0;
end
if nargin < 5
  prot = 0;
end
if roll ~=0
  c = cos(-roll);               % -sign gives TRANSPORT convention
  s = sin(-roll);
  O = [ c  0  s  0  0  0
        0  c  0  s  0  0
       -s  0  c  0  0  0
        0 -s  0  c  0  0
        0  0  0  0  1  0
        0  0  0  0  0  1];
else
  O = eye(6,6);
end

if L == 0
  R = O;
  return
end

h = angle/L;
kx2 = (k1+h*h);
ky2 = -k1;

if prot
  if angle == 0
    error('Do not call R_GEN6 with pole face rotation and no bend angle')
  end
  Rpr = eye(6,6);
  Rpr(2,1) = tan(angle/2)*h;
end

% horizontal plane first:
% ======================

kx   = sqrt(abs(kx2));
phix = kx*L;
if abs(phix) < 1E-12
  Rx = [1 L
        0 1];

  Dx = zeros(2,2);
  R56 = 0;
else
  if kx2>0
    co = cos(phix);
    si = sin(phix);
    Rx = [     co  si/kx
           -kx*si  co   ];
  else
    co = cosh(phix);
    si = sinh(phix);
    Rx = [     co  si/kx
            kx*si  co  ];
  end
  Dx = [0 h*(1-co)/kx2
        0 h*si/kx ];
  R56 = -(h^2)*(phix-kx*Rx(1,2))/(kx^3);
end


% vertical plane:
% ==============

ky   = sqrt(abs(ky2));
phiy = ky*L;
if abs(phiy) < 1E-12
  Ry = [1 L
        0 1];
else
  if ky2>0
    co = cos(phiy);
    si = sin(phiy);
    Ry = [     co  si/ky ;
           -ky*si  co   ];
  else
    co = cosh(phiy);
    si = sinh(phiy);
    Ry = [     co  si/ky;
            ky*si  co  ];
  end
end

R          = zeros(6,6);
R(1:2,1:2) = Rx;
R(3:4,3:4) = Ry;
R(1:2,5:6) = Dx;
R(5,1)     = -Dx(2,2);
R(5,2)     = -Dx(1,2);
R(5,5)     = 1;
R(5,6)     = R56;
R(6,6)     = 1;

R          = O*R*O';

if prot
  R = Rpr*R*Rpr;
end
