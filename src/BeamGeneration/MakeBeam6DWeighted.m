function beamout = MakeBeam6DWeighted( Initial, nrays, nsigma, allsame )
%
% MAKEBEAM6DWEIGHTED Create a beam with with rays uniformly randomly
% distributed on the interval [-nsigma:nsigma] with charge weighting
% according to a normal distribution function
%
% beamout = MakeBeam6DWeighted( Initial, nrays, nsigma, allsame ) returns a
%    beam in which the bunches have been populated with rays which are
%    uniformly randomly distributed in each degree of freedom and have charges
%    assigned according to a normal probability distribution.
%    Argument nrays is either a scalar (number of rays per bunch) or a
%    vector (number of rays in each bunch); nsigma gives the range over which
%    particles are distributed (either a scalar or a vector of length 6).  If
%    argument allsame == 0, a unique set of rays is generated for each
%    bunch, whereas if allsame == 1 the rays generated for bunch 1 are
%    reused for each subsequent bunch.  If allsame == 1, then all bunches
%    will have nrays(1) rays regardless of the value of nrays(2:end).
%
% See also:  MakeBeam6DGauss, MakeBeam6DSparse, MakeBeamPZGrid, MakeBeamPZGauss,
%            CreateBlankBeam, InitCondStruc.
%
% Version date:  21-March-2014
%==========================================================================

% get a blank beam with the correct # of bunches etc
beamout = CreateBlankBeam(Initial.NBunch,1,1,Initial.BunchInterval) ;

% get useful stuff out of the arguments
if length(nsigma)==1
  nsigma=ones(6,1).*nsigma;
end
nrays = nrays(:) ;
if (length(nrays) == Initial.NBunch)
  nraygen = nrays ;
else
  nraygen = nrays(1) * ones(Initial.NBunch,1) ;
end
%  centroid = [Initial.x.pos - Initial.x.Twiss.eta  ; ...
%              Initial.x.ang - Initial.x.Twiss.etap ; ...
%              Initial.y.pos - Initial.y.Twiss.eta  ; ...
%              Initial.y.ang - Initial.y.Twiss.etap ; ...
%              Initial.zpos  ; Initial.Momentum    ] ;
centroid = [Initial.x.pos   ; ...
  Initial.x.ang  ; ...
  Initial.y.pos   ; ...
  Initial.y.ang  ; ...
  Initial.zpos  ; Initial.Momentum    ] ;
relgam0 = sqrt(Initial.Momentum^2 + (0.000510998918)^2) / 0.000510998918 ;
emitx = Initial.x.NEmit / relgam0 ;
emity = Initial.y.NEmit / relgam0 ;

% compute uncorrelated sigmas

sig(1)  = sqrt(emitx*Initial.x.Twiss.beta) ;
sig(2)  = sqrt(emitx/Initial.x.Twiss.beta) ;
sig(3)  = sqrt(emity*Initial.y.Twiss.beta) ;
sig(4)  = sqrt(emity/Initial.y.Twiss.beta) ;
sig(5)  = Initial.sigz ;
sig(6)  = Initial.SigPUncorrel ;

% construct a correlation matrix

r = eye(6) ;
r(1,6) =   Initial.x.Twiss.eta / Initial.Momentum ;
r(2,1) = - Initial.x.Twiss.alpha / Initial.x.Twiss.beta ;
r(2,6) =   Initial.x.Twiss.etap / Initial.Momentum ;
r(3,6) =   Initial.y.Twiss.eta / Initial.Momentum ;
r(4,3) = - Initial.y.Twiss.alpha / Initial.y.Twiss.beta ;
r(4,6) =   Initial.y.Twiss.etap / Initial.Momentum ;
r(6,5) =   Initial.PZCorrel ;

% loop over bunches
for bunchno = 1:Initial.NBunch
  % if this is not the first bunch, AND we want allsame, just glue the old
  % bunch into the new bunch slots
  if ( (bunchno>1) && (allsame==1) )
    beamout.Bunch(bunchno) = beamout.Bunch(1) ;
  else
    beamout.Bunch(bunchno).Q = Initial.Q / nraygen(bunchno) * ...
      ones(1,nraygen(bunchno)) ;
    beamout.Bunch(bunchno).stop = zeros(1,nraygen(bunchno)) ;
    % construct the position vector one row at a time
    x = zeros(6,nraygen(bunchno)) ;
    offset = x ;
    ov = ones(1,nraygen(bunchno)) ;
    pdf=zeros(6,nraygen(bunchno));
    for count = 1:6
      % Create a uniformly distributed random vector of ray coordinates
      % from -nsigma to nsigma
      v = nsigma(count).*(2.*rand(1,nraygen(bunchno))-1) ;
      x(count,:) = sig(count) * v ;
      offset(count,:) = centroid(count)*ov ;
      pdf(count,:)=(1/sqrt(2*pi)).*exp(-v.^2./2);
    end
    % Weight each particle in bunch according to PDF (assign a charge to
    % each macro-particle such that the correct distributions are
    % represented)
    pdf=prod(pdf);
    pdf=pdf./sum(pdf);
    beamout.Bunch(bunchno).Q = Initial.Q .* pdf ;
    % put in the correlation and take out necessary offsets
    x = r*x + offset ;
    beamout.Bunch(bunchno).x = x ;
  end
end
beamout = CheckBeamMomenta(beamout) ;