function varargout=get_ATF2_corrs(opCode,energy,varargin)
%
% KL=get_ATF2_corrs(1,energy,Icorr,Ibend,fflag);
%
% Compute ATF2 corrector kicks (KL=BL/brho) from currents
%
% Icorr=get_ATF2_corrs(2,energy,KL,Ibend,fflag);
%
% Compute ATF2 corrector currents (KL=BL/brho) from kicks
%
% INPUTs (opCode=1):
%
%   opCode = 1 (I to KL conversion)
%   energy = energy (GeV) [scalar]
%   Icorr  = ATF2 corrector currents (amps) [29 element array]
%   Ibend  = ATF2 bend currents (amps) [11 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% INPUTs (opCode=2):
%
%   opCode = 2 (KL to I conversion)
%   energy = energy (GeV) [scalar]
%   KL     = corrector kicks (radians) [29 element array]
%   Ibend  = ATF2 bend currents (amps) [11 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% The order of correctors in the Icorr or KL arrays must be:
%
%   ZH100RX,ZH101RX,
%   ZX1X,ZH1X,ZH2X,ZX2X,ZX3X,ZH3X,ZH4X,ZH5X,ZH6X,ZH7X,ZH8X,ZH9X,ZH10X,
%   ZH1FF           
%   ZV100RX,
%   ZV1X,ZV2X,ZV3X,ZV4X,ZV5X,ZV6X,ZV7X,ZV8X,ZV9X,ZV10X,ZV11X,
%   ZV1FF           
%
% NOTE: the Ibend array is required in order to handle the trim-coil correctors
%       (ZX1X, ZX2X, and ZX3X)
%
% The order of bendss in the Ibend array must be:
%
%   BH1R,BS1X,BS2X,BS3X,
%   BH1X,BH2X,BH3X,B5FF,B2FF,B1FF,BDUMP
%
% OUTPUT (opCode=1):
%
%   KL    = ATF2 corrector kicks (radians; same signs as Icorr)
%            [29 element array, same order as Icorr]
%
% OUTPUT (opCode=2):
%
%   Icorr = ATF2 corrector currents (amps; same signs as KL)
%            [29 element array, same order as KL]

% ==============================================================================
% 22-FEB-2009, M. Woodley
%    Do either I-to-KL and KL-to-I conversion
% ==============================================================================

Ncorr=29;
Nbend=11;

% check input args

if (~exist('opCode','var'))
  error('opCode argument not supplied')
end
if (~exist('energy','var'))
  error('energy argument not supplied')
else
  if (energy<=0)
    error('Bad energy value')
  end
end
if (opCode==1)
  if (nargin<4)
    error('At least 4 input arguments required for opCode=1')
  end
  Icorr=varargin{1};
  if (length(Icorr)~=Ncorr)
    error('Incorrect Icorr length')
  end
elseif (opCode==2)
  if (nargin<4)
    error('At least 4 input arguments required for opCode=2')
  end
  KL=varargin{1};
  if (length(KL)~=Ncorr)
    error('Incorrect KL length')
  end
else
  error('Invalid opCode')
end
Ibend=varargin{2};
if (length(Ibend)~=Nbend)
  error('Incorrect Ibend length')
end
if (nargin>4)
  fflag=varargin{3};
else
  fflag=0;
end
fudge=(fflag~=0);

% ==============================================================================
% ATF2 CORRECTOR MAGNET DATA
% ==============================================================================

% there are 6 types of ATF2 corrector:

%    1 = Tecno model 59521 (horizontal)
%    2 = Tecno model 59789 (horizontal)
%    3 = Tecno model 58283 (horizontal)
%    4 = NKK type CH (horizontal)
%    5 = ZV100R (vertical; modified Tecno model 58284)
%    6 = Tecno model 58284 (vertical)
%    7 = NKK type CV (horizontal and vertical)
%    8 = BSHI-H trim (horizontal)
%    9 = BSHI-C trim (horizontal)

% B vs I slopes and effective lengths
% (from: ATF$MAG:MAG_KI_Z_TECNO_ZH82982.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_ZH82983.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_H58283.FOR
%        ATF$MAG:MAG_KI_Z_NKK_CH.FOR
%        ATF$MAG:MAG_KI_ZV100R.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_V58284.FOR
%        ATF$MAG:MAG_KI_Z_NKK_CV.FOR)

cbvi=[12.5e-3;5.9e-3;102.8e-4;108.0e-4;8.5e-3;111.5e-4;112.0e-4]; % T/amp
cleff=[0.24955;0.1679;0.11455;0.11921;0.13874;0.128141;0.1248];   % m

% corrector types

ctype=[1;2;8;3;3;8;9;3;3;7;3;3;7;7;4;4;5;6;6;6;6;6;6;6;6;7;6;6;6];

% "Kubo" fudge factors
% (none available)

cfudge=zeros(Ncorr,1);

% ==============================================================================

% there are 2 types of ATF2 bend that we need to be concerned with here:

%    1 = Sumitomo Heavy Industries type "H"
%    2 = Sumitomo Heavy Industries type "C"

% MMS data for quad types (I is amp, G is T/m)
% (from: ATF$MAG:MAG_KI_B_SHI_H.FOR
%        ATF$MAG:MAG_KI_B_SHI_C.FOR)

bI=[ ...
    0.0,  0.0; ...
   50.0, 50.0; ...
  100.0,100.0; ...
  150.0,150.0; ...
  200.0,200.0; ...
  250.0,250.0; ...
  300.0,300.0; ...
  320.0,320.0; ...
  340.0,340.0; ...
  360.0,360.0; ...
  380.0,380.0; ...
  400.0,400.0; ...
  420.0,420.0; ...
  440.0,440.0; ...
  460.0,460.0; ...
  480.0,480.0; ...
  500.0,500.0; ...
];

bB=[ ...
  0.0000,0.0000; ...
  0.1390,0.1560; ...
  0.2764,0.3095; ...
  0.4122,0.4632; ...
  0.5470,0.6145; ...
  0.6806,0.7631; ...
  0.8133,0.9092; ...
  0.8655,0.9635; ...
  0.9168,1.0110; ...
  0.9658,1.0492; ...
  1.0100,1.0806; ...
  1.0490,1.1079; ...
  1.0810,1.1313; ...
  1.1085,1.1530; ...
  1.1335,1.1727; ...
  1.1566,1.1911; ...
  1.1750,1.2000; ...
];

bleff=[0.79771;1.34296]; % m
Nm=[72;80]; % # of turns on main coil
Nt=[20;20]; % # of turns on trim coil
btype=[0;0;1;0;0;1;2;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0];
blist=[0;0;5;0;0;6;7;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0];
bsign=[0;0;-1;0;0;1;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0];

% ==============================================================================

% compute rigidity

Cb=1e10/2.99792458e8; % kG-m/GeV
brho=Cb*energy;

% perform requested conversion

if (opCode==1)
  
% convert current to KL
  
  KL=zeros(Ncorr,1);
  for n=1:Ncorr
    t=ctype(n);
    tb=btype(n);
    if (tb~=0)
      Imms=bI(:,tb);
      Bmms=bB(:,tb);
      Imain=Ibend(blist(n));
      Bmain=interp1(Imms,Bmms,Imain,'linear');
      Itrim=Icorr(n);
      if (fudge),Itrim=Itrim/(1+cfudge(t));end
      Itot=Imain+bsign(n)*(Nt(tb)/Nm(tb))*Itrim;
      Btot=interp1(Imms,Bmms,Itot,'linear');
      Btrim=Btot-Bmain;
      BL=sign(Icorr(n))*abs(Btrim)*bleff(tb);
    else
      BL=Icorr(n)*cbvi(t)*cleff(t); % T
      if (fudge),BL=BL/(1+cfudge(t));end
    end
    KL(n)=(10*BL)/brho; % radian
  end
  varargout{1}=KL;
else
  
% convert current to KL
  
  Icorr=zeros(Ncorr,1);
  for n=1:Ncorr
    t=ctype(n);
    tb=btype(n);
    if (tb~=0)
      Imms=bI(:,tb);
      Bmms=bB(:,tb);
      Imain=Ibend(blist(n));
      Bmain=interp1(Imms,Bmms,Imain,'linear'); % T
      Btrim=0.1*brho*KL(n)/bleff(tb); % T
      if (fudge),Btrim=Btrim*(1+cfudge(t));end
      Btot=Bmain+bsign(n)*Btrim;
      Itot=interp1(Bmms,Imms,Btot,'linear');
      Itrim=(Itot-Imain)/(Nt(tb)/Nm(tb));
      Icorr(n)=sign(KL(n))*abs(Itrim);
    else
      B=0.1*brho*KL(n)/cleff(t); % T
      if (fudge),B=B*(1+cfudge(t));end
      Icorr(n)=B/cbvi(t); % amps
    end
  end
  varargout{1}=Icorr;
end

end