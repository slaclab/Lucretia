function varargout=get_DR_skews(opCode,energy,varargin)
%
% KL=get_DR_skews(1,energy,Iskew,fflag);
%
% Compute KL (GL/brho) for DR skew quadrupole trims from currents and energy
%
% Iskew=get_DR_skews(2,energy,KL,fflag);
%
% Compute DR skew quadrupole trim currents from KL (GL/brho) and energy
%
% INPUTs (opCode=1):
%
%   opCode = 1 (I to KL conversion)
%   energy = DR energy (GeV) [scalar]
%   Iskew  = DR skew trim currents (amps) [68 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% INPUTs (opCode=2):
%
%   opCode = 2 (KL to I conversion)
%   energy = DR energy (GeV) [scalar]
%   KL     = normalized integrated strengths (1/m; positive means R41<0)
%            [68 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% The order of skew quadrupole trims in the Iskew or KL arrays must be:
%
%   SF1R.1:34,SD1R.1:34
%
% OUTPUT (opCode=1):
%
%   KL    = DR skew KLs (1/m; same signs as Iskew)
%            [68 element array, same order as Iskew]
%
% OUTPUT (opCode=2):
%
%   Iskew = DR skew trim currents (amps; same signs as KL)
%            [68 element array, same order as KL]

% ==============================================================================
% 22-FEB-2009, M. Woodley
%    Do either I-to-KL and KL-to-I conversion
% ==============================================================================

Nskew=68;

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
  Iskew=varargin{1};
  if (length(Iskew)~=Nskew)
    error('Incorrect Iskew length')
  end
elseif (opCode==2)
  if (nargin<3)
    error('At least 3 input arguments required for opCode=2')
  end
  KL=varargin{1};
  if (length(KL)~=Nskew)
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

% compute skew strengths
% (NOTE: calibration is 0.001 /m/amp @ 1.28 GeV, per Kubo 03dec09)

Cb=1e10/2.99792458e8; % kG-m/GeV
E0=1.28;
brho0=Cb*E0;
glvi=0.001*brho0;

brho=Cb*energy;
sqfudge=zeros(Nskew,1);

% perform requested conversion

if (opCode==1)
  
% convert current to KL
  
  KL=glvi*Iskew/brho;
  if (fudge),KL=KL./(1+sqfudge);end
  varargout{1}=KL;
else
  
% convert KL to current
  
  if (fudge),KL=KL.*(1+sqfudge);end
  Iskew=brho*KL/glvi;
  varargout{1}=Iskew;
end

end
