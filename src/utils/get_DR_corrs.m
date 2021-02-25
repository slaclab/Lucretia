function varargout=get_DR_corrs(opCode,energy,varargin)
%
% KL=get_DR_corrs(1,energy,Icorr,fflag);
%
% Compute DR corrector kicks (KL=BL/brho) from currents
%
% Icorr=get_DR_corrs(2,energy,KL,fflag);
%
% Compute DR corrector currents (KL=BL/brho) from kicks
%
% INPUTs (opCode=1):
%
%   opCode = 1 (I to KL conversion)
%   energy = DR energy (GeV) [scalar]
%   Icorr  = DR corrector currents (amps) [101 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% INPUTs (opCode=2):
%
%   opCode = 2 (KL to I conversion)
%   energy = DR energy (GeV) [scalar]
%   KL     = corrector kicks (radians) [101 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% The order of correctorss in the Icorr or KL arrays must be:
%
%    ZH1R,...,ZH14R,ZH16R,...,ZH48R (ZH15R is missing)
%    ZV1R,...,ZV15R,ZV17R,...,ZV51R (ZV16R is missing)
%    ZH100R,ZH101R,ZH102R,ZV100R (specials)
%
% OUTPUT (opCode=1):
%
%   KL    = DR corrector kicks (radians; same signs as Icorr)
%            [101 element array, same order as Icorr]
%
% OUTPUT (opCode=2):
%
%   Icorr = DR corrector currents (amps; same signs as KL)
%            [101 element array, same order as KL]

% ==============================================================================
% 22-FEB-2009, M. Woodley
%    Do either I-to-KL and KL-to-I conversion
% ==============================================================================

Ncorr=101;

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
  Icorr=varargin{1};
  if (length(Icorr)~=Ncorr)
    error('Incorrect Icorr length')
  end
elseif (opCode==2)
  if (nargin<3)
    error('At least 3 input arguments required for opCode=2')
  end
  KL=varargin{1};
  if (length(KL)~=Ncorr)
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
% DR CORRECTOR MAGNET DATA
% ==============================================================================

% there are 6 types of DR corrector:

%    1 = NKK type CH (horizontal)
%    2 = Tecno model 58283 (horizontal)
%    3 = NKK type CV (vertical)
%    4 = Tecno model 58928 (vertical)
%    5 = Tecno model 58284 (vertical)
%    6 = Tecno model 82982 (horizontal)
%    7 = Tecno model 82983 (horizontal)
%    8 = ZV100R (vertical; modified)

% B vs I slopes and effective lengths
% (from: ATF$MAG:MAG_KI_Z_NKK_CH.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_H58283.FOR
%        ATF$MAG:MAG_KI_Z_NKK_CV.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_V58928.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_V58284.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_ZH82982.FOR
%        ATF$MAG:MAG_KI_Z_TECNO_ZH82983.FOR
%        ATF$MAG:MAG_KI_ZV100R.FOR)

cbvi=[108.0e-4;102.8e-4;112.0e-4;111.5e-4;102.0e-4;12.5e-3;5.9e-3;8.5e-3]; % T/amp
cleff=[0.11921;0.11455;0.1248;0.128141;0.172;0.24955;0.1679;0.13874];      % m

% corrector types
% (from: ATF$MAG:MAG_KI_MAIN.FOR)

ctype=[ ...
  1;1;1;1;1;1;1;1;1;2;2;2;2;2;  1;1;1;1;1;1;1;1;1;1; ... %ZH1-25
  1;1;1;1;1;1;1;1;2;2;2;2;2;2;1;1;1;1;1;1;1;1;1; ...     %ZH26-48
  3;3;3;3;3;3;3;3;3;4;4;4;4;4;5;  3;3;3;3;3;3;3;3;3; ... %ZV1-25
  3;3;3;3;3;3;3;3;5;5;5;4;4;4;4;4;5;3;3;3;3;3;3;3;3; ... %ZV26-50
  3; ...                                                 %ZV51
  6;7;3;8];                                              %ZH100-102,ZV100

% "Kubo" fudge factors
% (none available)

cfudge=zeros(Ncorr,1);

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
    B=Icorr(n)*cbvi(t); % T
    KL(n)=(10*B)*cleff(t)/brho; % radian
    if (fudge),KL(n)=KL(n)/(1+cfudge(t));end
  end
  varargout{1}=KL;
else
  
% convert current to KL
  
  Icorr=zeros(Ncorr,1);
  for n=1:Ncorr
    t=ctype(n);
    B=0.1*brho*KL(n)/cleff(t); % T
    if (fudge),B=B*(1+cfudge(t));end
    Icorr(n)=B/cbvi(t); % amps
  end
  varargout{1}=Icorr;
end

end