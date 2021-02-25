function varargout=get_ATF2_bends(opCode,energy,varargin)
%
% KL=get_ATF2_bends(1,energy,Ibend,fflag);
%
% Compute ATF2 dipole KLs from power supply currents and beam energy
%
% Ibend=get_ATF2_bends(2,energy,KL,fflag);
%
% Compute ATF2 dipole power supply currents from KLs and beam energy
%
% NOTE: this code does not handle the extraction septa or extraction kickers
%
% INPUTS (opCode=1):
%
%   opCode = 1 (I to KL conversion)
%   energy = beam energy (GeV)
%   Ibend  = ATF2 bend currents (amps) [7 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% INPUTS (opCode=2):
%
%   opCode = 2 (KL to I conversion)
%   energy = beam energy (GeV)
%   KL     = dipole KLs (radian; positive means bend to the right)
%             [7 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% The order of dipoles in the Ibend or KL arrays must be:
%
%   BH1X,BH2X,BH3X,B5FF,B2FF,B1FF,BDUMP
%
% OUTPUT (opCode=1):
%
%   KL    = dipole KLs (radian; positive means bend to the right)
%            [7 element array]
%
% OUTPUT (opCode=2):
%
%   Ibend = dipole currents (amps) [7 element array]

% ==============================================================================
% 01-MAR-2009, M. Woodley
%    Created
% ==============================================================================

NATF2b=7;

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
  if (nargin<3)
    error('At least 3 input arguments required for opCode=1')
  end
  Ibend=varargin{1};
  if (length(Ibend)~=NATF2b)
    error('Incorrect Ibend length')
  end
elseif (opCode==2)
  if (nargin<3)
    error('At least 3 input arguments required for opCode=2')
  end
  KL=varargin{1};
  if (length(KL)~=NATF2b)
    error('Incorrect KL length')
  end
else
  error('Invalid opCode')
end
if (nargin>3)
  fflag=varargin{2};
else
  fflag=0;
end
fudge=(fflag~=0);

% ==============================================================================
%
% ATF2 BEND MAGNET DATA
%
% ==============================================================================

% there are 3 types of ATF2 bend:

%   1 = Sumitomo Heavy Industries type H
%   2 = Sumitomo Heavy Industries type C
%   3 = IHEP DEA-D38L575

% MMS data for bend types (I is amp, G is T)
% (from: ATF$MAG:MAG_KI_B_SHI_H.FOR
%        ATF$MAG:MAG_KI_B_SHI_C.FOR
%        SLAC magnetic measurements data (C. Spencer))

nrow=17; % number of data values for SHI-H and SHI-C types

% get averaged DEA data

[bI,bB]=ATF2_DEAave(); % integrated field (T-m)
[nr,nc]=size(bI);
bI=[bI;zeros(nrow-nr,1)]; % fill with zeros to nrow rows
bB=[bB;zeros(nrow-nr,1)]; % fill with zeros to nrow rows

% add SHI-H and SHI-C data

bI=[[ ...
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
],bI];

bB=[[ ...
  0.0   ,0.0   ; ...
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
],bB];

Nmms=[nrow,nrow,nr];
bleff=[0.79771,1.34296,0.6128]; % NOTE: measured DEA leff = 0.6221
Nm=[72,80,36];
Nt=[20,20,0];

% NOTE: convert DEA integrated strengths to field for standard handling

bB(:,3)=bB(:,3)/bleff(3);

% bend types

btype=[1;1;2;3;3;3;2];

% polarities

bsgn=[1;-1;-1;-1;-1;-1;1];

% fudge factors

bfudge=zeros(NATF2b,1);

% ==============================================================================

% compute rigidity

Cb=1e9/2.99792458e8; % T-m/GeV
brho=Cb*energy; % T-m

% perform requested conversion
% (NOTE: polynomial evaluation is not presently used to convert current
%        to gradient ... linear interpolation of MMS data is used)

if (opCode==1)
  
% convert current to KL
  
  KL=zeros(NATF2b,1);
  for n=1:NATF2b
    nt=btype(n);
    if (Ibend(n)<=0) % unipolar power supplies ... negative I is invalid
      continue
    else
      Imms=bI(1:Nmms(nt),nt);
      Bmms=bB(1:Nmms(nt),nt);
      B=interp1(Imms,Bmms,Ibend(n),'linear');
      if (fudge),B=B/(1+bfudge(n));end
      B=B*bsgn(n);
      KL(n)=B*bleff(nt)/brho; % radian
    end
  end
  varargout{1}=KL;
else
  
% convert KL to current
  
  Ibend=zeros(NATF2b,1);
  for n=1:NATF2b
    nt=btype(n);
    if (KL(n)==0)
      continue
    elseif (sign(KL(n))~=bsgn(n))
      continue % unipolar power supplies
    else
      Imms=bI(1:Nmms(nt),nt);
      Bmms=bB(1:Nmms(nt),nt);
      B=brho*abs(KL(n))/bleff(nt); % T
      if (fudge),B=B*(1+bfudge(n));end
      Ibend(n)=interp1(Bmms,Imms,B,'linear');
    end
    varargout{1}=Ibend;
  end
end

end
function [I,BL]=ATF2_DEAave()
%
% [I,BL]=ATF2_DEAave();
%
% Generate average integrated strength versus current for ATF2 DEA dipoles
%
% OUTPUTs:
%
%   I  = average current values (amps)
%   BL = average integrated strength values (T-m)

% DEA MMS data from Cherrill Spencer' file SummaryDEADipoleIntB.dlVCurrent.xls

Imms=[ ...
%    #1       #2      #3
   74.8737, 74.978, 74.7237; ...
   85.8528, 84.97 , 85.1874; ...
   95.3214, 94.985, 95.1369; ...
  105.3024,104.983,105.0999; ...
  115.2948,114.976,115.0764; ...
  125.2719,124.966,125.0409; ...
  135.2361,134.976,134.9892; ...
  145.2123,144.973,144.9489; ...
  155.205 ,154.966,154.9266; ...
  165.189 ,164.98 ,164.8941; ...
  174.972 ,174.97 ,174.9624; ...
];
BLmms=[ ...
%  #1         #2       #3
  0.1069431 ,0.10729 ,0.10684176 ; ...
  0.1225926 ,0.121638,0.121771999; ...
  0.1360842 ,0.135888,0.13596253 ; ...
  0.1503006 ,0.150169,0.150162438; ...
  0.1645231 ,0.164433,0.1643728  ; ...
  0.1787202 ,0.178669,0.17856412 ; ...
  0.1929014 ,0.192718,0.192736778; ...
  0.2070942 ,0.206832,0.206920116; ...
  0.2212932 ,0.221066,0.22110597 ; ...
  0.2354658 ,0.235292,0.2352682  ; ...
  0.24934725,0.249631,0.249572424; ...
];

% do a linear fit to all of the data

coef=polyfit(reshape(Imms,[],1),reshape(BLmms,[],1),1);

% use the fitted linear coefficients to generate "average" data

I=[0,75:10:175]';
BL=polyval(coef,I);
BL(1)=0; % zero means zero

end
