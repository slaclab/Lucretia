function varargout=get_ATF2_sexts(opCode,energy,varargin)
%
% KL=get_ATF2_sexts(1,energy,Isext,fflag);
%
% Compute ATF2 sextupole KLs from power supply currents and beam energy
%
% Isext=get_ATF2_sexts(2,energy,KL,fflag);
%
% Compute ATF2 sextupole power supply currents from KLs and beam energy
%
% INPUTS (opCode=1):
%
%   opCode = 1 (I to KL conversion)
%   energy = beam energy (GeV)
%   Isext  = ATF2 sextupole currents (amps) [9 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% INPUTS (opCode=2):
%
%   opCode = 2 (KL to I conversion)
%   energy = beam energy (GeV)
%   KL     = sextupole KLs (1/m^2; MAD8 sign convention) [9 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% The order of sextupoles in the Isext or KL arrays must be:
%
%   SF6FF,SK4FF,SK3FF,SF5FF,SD4FF,SK2FF,SK1FF,SF1FF,SD1FF
%
% OUTPUT (opCode=1):
%
%   KL     = sextupole KLs (1/m^2; positive means "SF") [9 element array]
%
% OUTPUT (opCode=2):
%
%   Isext  = ATF2 sextupole currents (amps) [9 element array]

% ==============================================================================
% 06-NOV-2012, M. Woodley
%    There are now 4 FF skew sextupoles
% ==============================================================================

NATF2s=9;

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
  Isext=varargin{1};
  if (length(Isext)~=NATF2s)
    error('Incorrect Isext length')
  end
elseif (opCode==2)
  if (nargin<3)
    error('At least 3 input arguments required for opCode=2')
  end
  KL=varargin{1};
  if (length(KL)~=NATF2s)
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
% ATF2 SEXTUPOLE MAGNET DATA
%
% ==============================================================================

% there are 6 ATF2 normal sextupoles (data courtesy of Cherrill Spencer (SLAC)),
% and 4 skew sextupoles (data generated using dG/dI=0.33638 (T/m)/amp from
% Mika Masuzawa (KEK))

%    1 = SF6   (1.625SX3.53 #02 (SLC))
%    2 = SF5   (1.625SX3.53 #03 (SLC))
%    3 = SD4   (1.625SX3.53 #01 (SLC))
%    4 = SF1   (2.13S3.00 SX2 (FFTB))
%    5 = SD0   (2.13S3.00 SK2 (FFTB))
%    6 = SK1-4 (skew sextupole (KEKB))

% MMS data for sextupoles (I is amp, G is integrated strength, T/m)

sI=[ ...
%  SF6    SF5    SD4    SF1      SD0       SK1-4
   0    , 0    , 0    , 0      , 0      ,-20; ...
   0.398, 0.396, 0.397, 2.9621 , 2.9615 ,  0; ...
   0.829, 0.829, 0.83 , 5.95714, 5.9561 , 20; ...
   1.399, 1.399, 1.399, 8.9525 , 8.9508 ,  0; ...
   1.832, 1.832, 1.832,11.96446,11.96224,  0; ...
   2.301, 2.302, 2.302, 0      , 0      ,  0; ...
   4.972, 4.97 , 4.971, 0      , 0      ,  0; ...
   7.443, 7.442, 7.443, 0      , 0      ,  0; ...
   9.912, 9.909, 9.911, 0      , 0      ,  0; ...
  15.069,15.065,15.068, 0      , 0      ,  0; ...
  19.962,19.962,19.962, 0      , 0      ,  0; ...
  24.962,24.961,24.962, 0      , 0      ,  0; ...
  30.055,30.055,30.055, 0      , 0      ,  0; ...
  35.066,35.066,35.066, 0      , 0      ,  0; ...
  39.908,39.912,39.91 , 0      , 0      ,  0; ...
  45.059,45.063,45.061, 0      , 0      ,  0; ...
  49.999,50.004,50.002, 0      , 0      ,  0; ...
];      

sGL=[ ...
%   SF6         SF5         SD4        SF1      SD0      SK1-4   
    0        ,  0        ,  0        , 0      , 0      ,-6.7276; ...
    2.2498363,  1.9905675,  2.0174219, 8.67672, 8.59829, 0     ; ...
    3.1659143,  2.9116955,  2.9416777,17.585  ,17.47928, 6.7276; ...
    4.3772233,  4.126823 ,  4.1611131,26.46082,26.33882, 0     ; ...
    5.292689 ,  5.0464729,  5.0835106,35.31407,35.13744, 0     ; ...
    6.2995431,  6.0563734,  6.096839 , 0      , 0      , 0     ; ...
   12.1380489, 11.896674 , 11.9431827, 0      , 0      , 0     ; ...
   17.5544078, 17.3217305, 17.3880054, 0      , 0      , 0     ; ...
   23.0261077, 22.7888383, 22.8818389, 0      , 0      , 0     ; ...
   34.5769838, 34.3304156, 34.4413399, 0      , 0      , 0     ; ...
   45.6399439, 45.3629794, 45.5144949, 0      , 0      , 0     ; ...
   57.0404355, 56.6435797, 56.8469279, 0      , 0      , 0     ; ...
   68.6272702, 68.0544545, 68.3276361, 0      , 0      , 0     ; ...
   79.9906013, 79.2441692, 79.5503455, 0      , 0      , 0     ; ...
   90.9077093, 90.0244698, 90.3472626, 0      , 0      , 0     ; ...
  102.4637655,101.4447267,101.7951883, 0      , 0      , 0     ; ...
  113.4777391,112.3449583,112.7209785, 0      , 0      , 0     ; ...
];

Nmms=[17;17;17;5;5;3];

% sextupole types and polarities (zero = bipolar)

stype=[1,6,6,2,3,6,6,4,5];
ssgn=[1,0,0,-1,1,0,0,-1,1];

% "Kubo" fudge factors
% (none defined at this time ... )

sfudge=zeros(NATF2s,1);

% ==============================================================================

% compute rigidity

Cb=1e10/2.99792458e8; % kG-m/GeV
brho=Cb*energy;

% perform requested conversion
% (NOTE: polynomial evaluation is not presently used to convert current
%        to gradient ... linear interpolation of MMS data is used)

if (opCode==1)
  
% convert current to KL
  
  KL=zeros(NATF2s,1);
  for n=1:NATF2s
    bipolar=(ssgn(n)==0);
    if ((Isext(n)>0)||bipolar)
      nt=stype(n);
      Imms=sI(1:Nmms(nt),nt);
      GLmms=sGL(1:Nmms(nt),nt);
      GL=interp1(Imms,GLmms,Isext(n),'linear');
      if (fudge),GL=GL/(1+sfudge(n));end
      KL(n)=(10*GL)/brho;
      if (~bipolar)
        KL(n)=ssgn(n)*KL(n);
      end
    end
  end
  varargout{1}=KL;
else
  
% convert KL to current
  
  Isext=zeros(NATF2s,1);
  for n=1:NATF2s
    bipolar=(ssgn(n)==0);
    if (KL(n)==0)
      continue
    elseif (~bipolar&&(sign(KL(n))~=ssgn(n)))
      continue
    else
      nt=stype(n);
      Imms=sI(1:Nmms(nt),nt);
      GLmms=sGL(1:Nmms(nt),nt);
      GL=0.1*brho*abs(KL(n));
      if (fudge),GL=GL*(1+sfudge(n));end
      Isext(n)=interp1(GLmms,Imms,GL,'linear');
    end
  end
  varargout{1}=Isext;
end

end
