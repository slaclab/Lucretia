function varargout=DR_energy(opCode,varargin)
%
% energy=DR_energy(1,Ibh1r)
%
% Compute DR energy from BH1R current
%
% Ibh1r=DR_energy(2,energy)
%
% Compute BH1R current from DR energy
%
% INPUTs (opCode=1):
%
%   opCode = 1 (I to energy conversion)
%   Ibh1r  = BH1R current (amp) [scalar]
%
% INPUTs (opCode=2):
%
%   opCode = 2 (energy to I conversion)
%   energy = DR energy (GeV) [scalar]
%
% OUTPUT (opCode=1):
%
%   energy = computed DR energy (GeV) assuming bend angle of pi/18
%
% OUTPUT (opCode=2):
%
%   Ibh1r  = BH1R current (amp) assuming bend angle of pi/18

% ==============================================================================
% 24-OCT-2012, M. Woodley
%    New BH1R energy fudge per ATF2 eLog: 812.801 amp = 1.269 GeV
% 07-FEB-2010, M. Woodley
%    New BH1R energy fudge per T. Okugi: 813.87 amp = 1.2820 GeV
% 25-JAN-2005, M. Woodley
%    The BH1R power supply was modified ... amps are no longer what they
%    used to be; apply a fudge on the current so that 812.17 amps
%    corresponds to 1.2800 GeV
% ==============================================================================

persistent f

% check input args

if (~exist('opCode','var'))
  error('opCode argument not supplied')
end
if (opCode==1)
  if (nargin<2)
    error('2 input arguments required for opCode=1')
  end
  Ibh1r=varargin{1};
  if (Ibh1r<=0)
    error('Invalid Ibh1r')
  end
elseif (opCode==2)
  if (nargin<2)
    error('2 input arguments required for opCode=2')
  end
  energy=varargin{1};
  if (energy<=0)
    error('Bad energy value')
  end
else
  error('Invalid opCode')
end

% pre-computed "ivb" polynomial for BH1R (kG-m = f(amps))

bvibh1=[ 4.0507212e-03, 9.3415182e-03, 9.9991137e-07, ...
        -2.6931190e-09, 2.7155309e-12,-1.0346054e-15];

% some constants

Cb=1e10/2.99792458e8; % kG-m/GeV
theta=pi/18;

% fudge factor

if (isempty(f))
  I0=812.801; % amp ... from God
  E0=1.269; % GeV ... from God
  BL0=(Cb*E0)*theta; % kG-m
  c=fliplr(bvibh1);
  c(end)=c(end)-BL0;
  r=roots(c);
  r=r(imag(r)==0);
  id=(r>0)&(r<1000); % assume that 0 < Ibh1r < 1000 amps
  f=r(id)/I0;
end

% perform requested conversion

if (opCode==1)
  
% convert current to energy

  BL=polyval(fliplr(bvibh1),f*Ibh1r);
  energy=BL/(Cb*theta);
  varargout{1}=energy;
else
  
% convert energy to current

  BL=(Cb*energy)*theta; % kG-m
  c=fliplr(bvibh1);
  c(end)=c(end)-BL;
  r=roots(c);
  r=r(imag(r)==0);
  id=(r>0)&(r<1000); % assume that 0 < Ibh1r < 1000 amps
  Ibh1r=r(id)/f;
  varargout{1}=Ibh1r;
end

end

