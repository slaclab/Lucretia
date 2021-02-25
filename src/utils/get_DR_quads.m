function varargout=get_DR_quads(opCode,energy,varargin)
%
% [Iquad,KL]=get_DR_quads(1,energy,Imain,Itrim,fflag);
%
% Compute "effective" main coil currents and normalized integrated strengths
% (GL/brho) for individual DR quadrupoles from main coil currents, trim coil
% currents, and beam energy
%
% Iquad=get_DR_quads(2,energy,KL,fflag);
%
% Compute main coil currents for DR quadrupole families from normalized
% integrated strengths (GL/brho) and beam energy
%
% INPUTs (opCode=1):
%
%   opCode = 1 (I to KL conversion)
%   energy = DR energy (GeV) [scalar]
%   Imain  = DR quadrupole family main coil currents (amps) [26 element array]
%   Itrim  = DR quadrupole trim coil currents (amps) [100 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% INPUTs (opCode=2):
%
%   opCode = 2 (KL to I conversion)
%   energy = DR energy (GeV) [scalar]
%   KL     = normalized integrated strengths (1/m; positive means "QF")
%             [26 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% The order of quadrupole families in the Imain or KL arrays must be:
%
%   QF1R,QF2R,
%   QM1R,QM2R,QM3R,QM4R,QM5R,
%   QM6R,QM7R,QM8R,QM9R,QM10R,
%   QM11R,QM12R,QM13R,QM14R,QM15R,
%   QM16R,QM17R,QM18R,QM19R,QM20R,
%   QM21R,QM22R,QM23R,QM7AR
%
% The order of quadrupole trims in the Itrim array must be:
%
%   QF1R.1:28,QF2R.1:26,
%   QM1R.1:2,QM2R.1:2,QM3R.1:2,QM4R.1:2,QM5R.1:2,
%   QM6R.1:2,QM7R.1:2,QM8R.1:2,QM9R.1:2,QM10R.1:2,
%   QM11R.1:2,QM12R.1:2,QM13R.1:2,QM14R.1:2,QM15R.1:2,
%   QM16R.1:2,QM17R.1:2,QM18R.1:2,QM19R.1:2,QM20R.1:2,
%   QM21R.1:2,QM22R.1:2,QM23R.1:2
%
% OUTPUTs (opCode=1):
%
%   Iquad = effective DR quadrupole currents (amps) [100 element array,
%            same order as Itrim]
%   KL    = normalized integrated strengths (1/m; positive means "QF")
%            [26 element array]
%
% OUTPUT (opCode=2):
%
%   Iquad = DR quadrupole family currents (amps) [26 element array]

% ==============================================================================
% 10-DEC-2010, M. Woodley
%    Switch polarity of QM2R (was QD) ... design model wants it QF (turned OFF
%    for normal operations)
% 26-FEB-2009, M. Woodley
%    Correct Nm(=49) and Nt(=20) for QEA quads per IHEP drawing
% 22-FEB-2009, M. Woodley
%    Add support for QM7AR and QEAs; do either I-to-KL and KL-to-I conversion
% 22-MAY-2007, M. Woodley
%    Data for IHEP "QEA" quads (QM12R, QM13R, QM14R) from control system file
% 06-FEB-2007, M. Woodley
%    Tokin 3581 quads replaced with IHEP "QEA" quads (QM12R, QM13R, QM14R)
% ==============================================================================

Nmain=26;
Ntrim=100;

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
  Imain=varargin{1};
  if (length(Imain)~=Nmain)
    error('Incorrect Imain length')
  end
  Itrim=varargin{2};
  if (length(Itrim)~=Ntrim)
    error('Incorrect Itrim length')
  end
  if (nargin>4)
    fflag=varargin{3};
  else
    fflag=0;
  end
elseif (opCode==2)
  if (nargin<3)
    error('At least 3 input arguments required for opCode=2')
  end
  KL=varargin{1};
  if (length(KL)~=Nmain)
    error('Incorrect KL length')
  end
  if (nargin>3)
    fflag=varargin{2};
  else
    fflag=0;
  end
else
  error('Invalid opCode')
end
fudge=(fflag~=0);

% ==============================================================================
% DR QUAD MAGNET DATA
% ==============================================================================

% there are 11 DR quadrupole types:

%    1 = Hitachi type 1
%    2 = Hitachi type 2
%    3 = Hitachi type 3
%    4 = Hitachi type 4
%    5 = Tokin type 3325 ("18 cm")
%    6 = Tokin type 3582 ("18 cm (42p)")
%    7 = Tokin type 3393 ("6 cm")
%    8 = Tokin type 3581 ("6 cm")
%    9 = QEA-13 (IHEP type D32L180 # 13) ... in series with QEA-14
%   10 = QEA-19 (IHEP type D32L180 # 19) ... in series with QEA-17
%   11 = QEA-21 (IHEP type D32L180 # 21) ... in series with QEA-18

% magnet data  (I is amp, G is T/m, leff is m)
% (from: ATF$MAG:MAG_KI_Q_HITACHI_1.FOR
%        ATF$MAG:MAG_KI_Q_HITACHI_2.FOR
%        ATF$MAG:MAG_KI_Q_HITACHI_3.FOR
%        ATF$MAG:MAG_KI_Q_HITACHI_4.FOR
%        ATF$MAG:MAG_KI_Q_TOKIN_3325.FOR
%        ATF$MAG:MAG_KI_Q_TOKIN_3582.FOR
%        ATF$MAG:MAG_KI_Q_TOKIN_3393.FOR
%        ATF$MAG:MAG_KI_Q_TOKIN_3581.FOR
%        mainamp50AmaxAbs.txt (M.Masuzawa))

Nmms=[11,11,11,11,11,11,8,11,16,16,16];

qI=[ ...
    0.0,  0.0,  0.0,  0.0,  0.00,  0.0,  0.0,  0.0,0.0000000,0.0000000,0.0000000; ...
   14.2, 10.2, 20.4, 20.2,100.00, 50.0, 20.0, 50.0,3.3333001,3.3333001,3.3333001; ...
   28.6, 20.2, 40.0, 40.2,200.00,100.0, 40.0, 80.0,6.6666999,6.6666999,6.6666999; ...
   42.2, 30.2, 60.0, 60.4,300.00,150.0, 60.0,100.0,10.000000,10.000000,10.000000; ...
   56.0, 40.0, 80.2, 80.4,400.00,170.0, 80.0,130.0,13.333000,13.333000,13.333000; ...
   70.2, 50.0,100.8,100.0,417.24,190.0,100.0,150.0,16.667000,16.667000,16.667000; ...
   84.2, 60.2,120.2,120.2,434.48,210.0,120.0,170.0,20.000000,20.000000,20.000000; ...
   98.2, 70.2,140.0,140.2,451.72,220.0,139.0,190.0,23.333000,23.333000,23.333000; ...
  112.2, 80.2,160.2,160.2,468.97,230.0,  0  ,210.0,26.667000,26.667000,26.667000; ...
  126.0, 90.2,180.2,180.0,503.45,240.0,  0  ,230.0,30.000000,30.000000,30.000000; ...
  140.2,100.2,200.2,200.4,512.07,245.0,  0  ,245.0,33.333000,33.333000,33.333000; ...
    0  ,  0  ,  0  ,  0  ,  0   ,  0  ,  0  ,  0  ,36.667000,36.667000,36.667000; ...
    0  ,  0  ,  0  ,  0  ,  0   ,  0  ,  0  ,  0  ,40.000000,40.000000,40.000000; ...
    0  ,  0  ,  0  ,  0  ,  0   ,  0  ,  0  ,  0  ,43.333000,43.333000,43.333000; ...
    0  ,  0  ,  0  ,  0  ,  0   ,  0  ,  0  ,  0  ,46.667000,46.667000,46.667000; ...
    0  ,  0  ,  0  ,  0  ,  0   ,  0  ,  0  ,  0  ,50.000000,50.000000,50.000000; ...
];                                                 

qG=[ ...
   0.000, 0.000, 0.000, 0.000, 0.00, 0.0, 0.0, 0.0,0         ,0         ,0         ; ...
   2.350, 3.832, 5.628, 4.612,10.80, 7.6, 3.4, 7.5,0.32538533,0.33133936,0.32236442; ...
   4.712, 7.646,11.109, 9.201,21.45,14.9, 6.8,11.8,0.63208896,0.63420987,0.62648714; ...
   6.981,11.498,16.656,13.858,32.05,22.2,10.0,14.8,0.94148004,0.93937200,0.93356729; ...
   9.256,15.202,22.243,18.440,42.65,25.0,13.2,19.1,1.2529521 ,1.2468803 ,1.2427585 ; ...
  11.604,18.927,27.866,22.884,44.70,28.0,16.6,21.8,1.5639043 ,1.5545483 ,1.5514039 ; ...
  13.892,22.657,33.122,27.464,46.65,30.8,19.8,24.4,1.8741443 ,1.8627108 ,1.8593891 ; ...
  16.167,26.285,38.487,31.925,48.10,32.1,23.0,26.4,2.1841798 ,2.1716371 ,2.1672122 ; ...
  18.410,29.896,43.828,36.386,49.70,33.3, 0  ,28.0,2.5015478 ,2.4888871 ,2.4827976 ; ...
  20.629,33.438,48.575,40.726,52.35,34.2, 0  ,29.3,2.8104367 ,2.7979240 ,2.7898581 ; ...
  22.898,36.758,51.971,45.143,52.80,34.7, 0  ,30.1,3.1178350 ,3.1058538 ,3.0956228 ; ...
   0    , 0    , 0    , 0    , 0   , 0  , 0  , 0  ,3.4247386 ,3.4134352 ,3.4008279 ; ...
   0    , 0    , 0    , 0    , 0   , 0  , 0  , 0  ,3.7313569 ,3.7210751 ,3.7056735 ; ...
   0    , 0    , 0    , 0    , 0   , 0  , 0  , 0  ,4.0364571 ,4.0274525 ,4.0088239 ; ...
   0    , 0    , 0    , 0    , 0   , 0  , 0  , 0  ,4.3408098 ,4.3334475 ,4.3110003 ; ...
   0    , 0    , 0    , 0    , 0   , 0  , 0  , 0  ,4.6463265 ,4.6411600 ,4.6146355 ; ...
];

qleff=[0.078765,0.07867,0.19847,0.198745,0.19886,0.202628,0.07890677,0.084339,0.19849,0.19849,0.19849];

Nm=[17,39,29,24,11,26,17,26,49,49,49]; % main coil turns
Nt=[20,20,20,20,20,20,20,20,20,20,20]; % trim coil turns

% NOTE: data for types 9-11 (IHEP type D32L180 ("QEA")) are integrated strength
%       in Tesla ... divide by leff for standard handling

for n=9:11
  qG(:,n)=qG(:,n)/qleff(n);
end

% quadrupole family names

qname=['QF1R ';'QF2R ';'QM1R ';'QM2R ';'QM3R ';'QM4R ';'QM5R ';'QM6R '; ...
       'QM7R ';'QM8R ';'QM9R ';'QM10R';'QM11R';'QM12R';'QM13R';'QM14R'; ...
       'QM15R';'QM16R';'QM17R';'QM18R';'QM19R';'QM20R';'QM21R';'QM22R'; ...
       'QM23R';'QM7AR'];

% quadrupole family types
% (from: ATF$MAG:MAG_KI_MAIN.FOR)

qtype=[1;5;3;2;2;3;4;4;7;7;4;4;6;9;10;11;6;4;4;4;4;3;2;2;3;8];

% polarities

qsgn=[1;1;1;-1;-1;1;1;-1;1;1;-1;1;-1;1;-1;1;-1;1;-1;-1;1;1;-1;-1;1;1];
qsgn(4)=+1; % QM2R

% "Kubo" fudge factors
% (NOTE: set fudge factors to zero ... who knows what they are now)

qfudge=[ ...
  0.0; ... % QF1R
  0.0; ... % QF2R
  0.0; ... % QM1R
  0.0; ... % QM2R
  0.0; ... % QM3R
  0.0; ... % QM4R
  0.0; ... % QM5R
  0.0; ... % QM6R
  0.0; ... % QM7R
  0.0; ... % QM8R
  0.0; ... % QM9R
  0.0; ... % QM10R
  0.0; ... % QM11R
  0.0; ... % QM12R
  0.0; ... % QM13R
  0.0; ... % QM14R
  0.0; ... % QM15R
  0.0; ... % QM16R
  0.0; ... % QM17R
  0.0; ... % QM18R
  0.0; ... % QM19R
  0.0; ... % QM20R
  0.0; ... % QM21R
  0.0; ... % QM22R
  0.0; ... % QM23R
  0.0; ... % QM7AR
];

% quadrupole family pointers

idq=[ ...
   1*ones(28,1); ... % QF1R
   2*ones(26,1); ... % QF2R
   3*ones( 2,1); ... % QM1R
   4*ones( 2,1); ... % QM2R
   5*ones( 2,1); ... % QM3R
   6*ones( 2,1); ... % QM4R
   7*ones( 2,1); ... % QM5R
   8*ones( 2,1); ... % QM6R
         [26;9]; ... % QM7AR,QM7R
  10*ones( 2,1); ... % QM8R
  11*ones( 2,1); ... % QM9R
  12*ones( 2,1); ... % QM10R
  13*ones( 2,1); ... % QM11R
  14*ones( 2,1); ... % QM12R
  15*ones( 2,1); ... % QM13R
  16*ones( 2,1); ... % QM14R
  17*ones( 2,1); ... % QM15R
  18*ones( 2,1); ... % QM16R
  19*ones( 2,1); ... % QM17R
  20*ones( 2,1); ... % QM18R
  21*ones( 2,1); ... % QM19R
  22*ones( 2,1); ... % QM20R
  23*ones( 2,1); ... % QM21R
  24*ones( 2,1); ... % QM22R
  25*ones( 2,1); ... % QM23R
];

% ==============================================================================

% compute rigidity

Cb=1e10/2.99792458e8; % kG-m/GeV
brho=Cb*energy;

% perform requested conversion
% (NOTE: polynomial evaluation is not presently used to convert current to
%        gradient ... linear interpolation of MMS data is used)

if (opCode==1)
  
% convert current to KL
  
  Iquad=zeros(Ntrim,1);
  KL=zeros(Ntrim,1);
  for n=1:Ntrim
    m=idq(n); % quad family (1-26)
    t=qtype(m); % quad type (1-11)
    Iquad(n)=Imain(m)+(Nt(t)/Nm(t))*Itrim(n);
    if (Iquad(n)<=0)
      G=0;
    else
      G=interp1(qI(1:Nmms(t),t),qG(1:Nmms(t),t),Iquad(n),'linear');
    end
    KL(n)=qsgn(m)*(10*G)*qleff(t)/brho;
    if (fudge),KL(n)=KL(n)/(1+qfudge(m));end
  end
  varargout{1}=Iquad;
  varargout{2}=KL;
else
  
% convert KL to current
  
  Iquad=zeros(Nmain,1);
  for n=1:Nmain
    if (KL(n)~=0)
      t=qtype(n); % quad type (1-11)
      G=0.1*brho*abs(KL(n))/qleff(t); % T/m
      if (fudge),G=G*(1+qfudge(t));end
      Iquad(n)=interp1(qG(1:Nmms(t),t),qI(1:Nmms(t),t),G,'linear');
    end
  end
  varargout{1}=Iquad;
end

end
