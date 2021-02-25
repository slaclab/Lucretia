function beamout = MakeBeam6DGauss( Initial, nrays, nsigma, allsame )
%
% MAKEBEAM6DGAUSS Create a beam with Gaussian-generated rays according to a
% desired initial distribution.
%
% beamout = MakeBeam6DGauss( Initial, nrays, nsigma, allsame ) returns a
%    beam in which the bunches have been populated with rays which are
%    Gaussian distributed in each degree of freedom and have equal charges.
%    Argument nrays is either a scalar (number of rays per bunch) or a
%    vector (number of rays in each bunch); nsigma is the truncation factor
%    for the distribution (either a scalar or a vector of length 6).  If
%    argument allsame == 0, a unique set of rays is generated for each
%    bunch, whereas if allsame == 1 the rays generated for bunch 1 are
%    reused for each subsequent bunch.  If allsame == 1, then all bunches
%    will have nrays(1) rays regardless of the value of nrays(2:end).
%
% See also:  MakeBeam6DSparse, MakeBeamPZGrid, MakeBeamPZGauss,
%            MakeBeam6DWeighted, CreateBlankBeam, InitCondStruc.
%
% Version date:  30-Jun-2016.

% MOD:
%      30-Jun-2016, GW:
%         Add ability to specify coupling parameters in Initial.CoupCoef
%      27-Jan-2011, GW:
%         Don't try and generate gaussian spread if just asking for single
%         ray, just populate bunch centroid
%      24-may-2007, PT:
%         check for bad transverse and total momentum before
%         returning the beam to the caller.
%      PT, 23-jun-2006:
%        bugfix:  incorrect inclusion of correlations when there is an
%        initial dispersion.

%==========================================================================

% get a blank beam with the correct # of bunches etc

beamout = CreateBlankBeam(Initial.NBunch,1,1,Initial.BunchInterval) ;

% get useful stuff out of the arguments

nsigma = nsigma(:) ;
if (length(nsigma) == 6)
  trunc = nsigma ;
else
  trunc = ones(6,1) * nsigma(1) ;
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
if isfield(Initial,'CoupCoef')
  r(3,1)= Initial.CoupCoef.xy / sqrt(sig(1)^2*sig(3)^2) ;
  r(3,2)= Initial.CoupCoef.xpy / sqrt(sig(2)^2*sig(3)^2) ;
  r(4,1)= Initial.CoupCoef.xyp / sqrt(sig(1)^2*sig(4)^2) ;
  r(4,2)= Initial.CoupCoef.xpyp / sqrt(sig(2)^2*sig(4)^2) ;
  if isfield(Initial.CoupCoef,'xz')
    r(1,5)= Initial.CoupCoef.xz / sqrt(sig(1)^2*sig(5)^2) ;
  end
end
r(isnan(r))=0;

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
    for count = 1:6
      
      v = randn(1,nraygen(bunchno)) ;
      
      % replace any entries which exceed the truncation limit
      % (don't do this if just asking for single ray)
      if length(v)>1
        vv = find(abs(v) > trunc(count)) ;
        for count2 = 1:length(vv)
          while (abs(v(vv(count2)))>trunc(count))
            v(vv(count2)) = randn(1) ;
          end
        end
      end
      
      x(count,:) = sig(count) * v ;
      offset(count,:) = centroid(count)*ov ;
      
    end
    
    % put in the correlation and take out necessary offsets
    % if just asking for single ray, just make this the centroid
    if length(v)>1
      r1 = eye(6) ; r1(6,5)=r(6,5); r(6,5)=0; % Add energy spread due to chirp first
      x = r1*x ;
      x = r*x + offset ;
      beamout.Bunch(bunchno).x = x ;
    else
      beamout.Bunch(bunchno).x = centroid ;
    end
    
  end
  
end

beamout = CheckBeamMomenta(beamout) ;
%