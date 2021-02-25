function varargout=get_DR_sexts(opCode,energy,varargin)
%
% KL=get_DR_sexts(1,energy,Isext,fflag);
%
% Compute KL (G'L/brho) for DR sextupole families from currents and energy
%
% Isext=get_DR_sexts(2,energy,KL,fflag);
%
% Compute DR sextupole family currents from KL (G'L/brho) and energy
%
% INPUTs (opCode=1):
%
%   opCode = 1 (I to KL conversion)
%   energy = DR energy (GeV) [scalar]
%   Isext  = DR sextupole family currents (amps) [2 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% INPUTs (opCode=2):
%
%   opCode = 2 (KL to I conversion)
%   energy = DR energy (GeV) [scalar]
%   KL     = normalized integrated sextupole strengths (1/m^2) [2 element array]
%   fflag  = (optional) if present and nonzero, fudge factors will be used
%
% The order of sextupole families in the Isext or KL arrays must be:
%
%    SF1R,SD1R
%
% OUTPUT (opCode=1):
%
%   KL    = DR normalized integrated sextupole strengths (1/m^2)
%            [2 element array]
%
% OUTPUT (opCode=2):
%
%   Isext = DR sextupole family currents (amps) [2 element array]

% ==============================================================================
% 10-DEC-2012, M. Woodley
%    Sextupole signs changed ... bending direction changed in Nov 2009
% 22-FEB-2009, M. Woodley
%    Do either I-to-KL and KL-to-I conversion
% ==============================================================================

Ndrs=2;

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
  if (length(Isext)~=Ndrs)
    error('Incorrect Isext length')
  end
elseif (opCode==2)
  if (nargin<3)
    error('At least 3 input arguments required for opCode=2')
  end
  KL=varargin{1};
  if (length(KL)~=Ndrs)
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
% DR SEXTUPOLE MAGNET DATA
% ==============================================================================

% there is only one type of DR sextupole (Mitsubishi)

% MMS data for sextupoles (I is amp, B is T @ r0 = 6 mm)
% (from: ATF$CTRL:[MAGNET.PRO.MST]MAG_KI_S_MELCO.FOR;2)

Nmms=17;

sI=[  0.00000;  0.07500; 25.01000; 49.98700; 75.00100; ...
    124.99800;149.99100;175.00000;200.00700;224.98399; ...
    249.98300;275.00299;299.98700;324.99500;350.00000; ...
    374.98999;399.99600];

sB=[  0.00000;  0.00050;  0.00986;  0.01959;  0.02927; ...
      0.04816;  0.05737;  0.06623;  0.07398;  0.07947; ...
      0.08339;  0.08645;  0.08891;  0.09093;  0.09261; ...
      0.09404;  0.09527];

r0=0.006;
sG=2*sB/r0^2; % T/m^2

% sextupole names

sname=['SF1R';'SD1R'];

% sextupole effective length
% (from: ATF$CTRL:[MAGNET.PRO.MST]MAG_KI_S_MELCO.FOR;2)

sleff=0.07077;

% SAD K2 signs

ssgn=[-1;1]; % [SF1R;SD1R]

% "Kubo" fudge factors (unknown whether such things exist ...)

sfudge=[0;0]; % [SF1R;SD1R]

% ==============================================================================

% compute rigidity

Cb=1e10/2.99792458e8; % kG-m/GeV
brho=Cb*energy;

% perform requested conversion
% (NOTE: polynomial evaluation is not presently used to convert current
%        to gradient ... linear interpolation of MMS data is used)

if (opCode==1)
  
% convert current to KL
  
  KL=zeros(Ndrs,1);
  for n=1:Ndrs
    if (Isext(n)>0)
      G=interp1(sI,sG,Isext(n),'linear');
      if (fudge),G=G/(1+sfudge(n));end
      KL(n)=ssgn(n)*(10*G)*sleff/brho;
    end
  end
  varargout{1}=KL;
else
  
% convert KL to current
  
  Isext=zeros(Ndrs,1);
  for n=1:Ndrs
    if (KL(n)==0)
      continue
    elseif (sign(KL(n))~=ssgn(n))
      continue
    else
      G=0.1*brho*abs(KL(n))/sleff; % T/m^2
      if (fudge),G=G*(1+sfudge(n));end
      Isext(n)=interp1(sG,sI,G,'linear');
    end
  end
  varargout{1}=Isext;
end

end
