function beamout = MakeBeamPZGauss( Initial, nsigmas, nray )

% MAKEBEAMPZGauss Generate Gaussian-distributed Lucretia beam in 
%    longitudinal phase space.
%
%    B = MakeBeamPZGauss( Initial, nsigmas, nray ) returns a 
%    Lucretia beam structure in which all the rays have the same transverse
%    coordinates but are randomly distributed according to a Gaussian
%    distribution function in the P-z (longitudinal) phase plane.
%    Argument Initial contains all initial conditions such as beam energies,
%    bunch lengths, etc (see function InitCondStruc for details).  Argument
%    nsigmas tells the # of sigmas on which to truncate.  Argument nray
%    tells the number of rays per bunch, and is a scalar. 
%
%    All of the rays in a given bunch have equal charge, and the sum of 
%    the charges is given by the values in Initial.
%
%    SEE ALSO:  MakeBeamPZGrid, MakeBeam6DSparse, MakeBeam6DGauss, MakeBeam6DWeighted
%               CreateBlankBeam, InitCondStruc.
%
% Version date:  24-May-2007.

% MOD:
%      24-may-2007, PT:
%         check for bad transverse and total momentum before
%         returning the beam to the caller.

%============================================================================

%
% very sketchy early version has virtually no exception checking!
% Compute the fractional charge per ray:
%
  nray = nray(1) ;
  Qper = Initial.Q / nray ;
%
% unpack # of sigmas
%
  [nsigz,nsigP] = deal(nsigmas) ;
%
% get a blank beam with the right number of rays
%
  beamout = CreateBlankBeam(Initial.NBunch, nray, Initial.Momentum, ...
                            Initial.BunchInterval) ;
%
% create an initial set of rays and charges which can be copied into the
% beam
%
  x = [Initial.x.pos * ones(1,nray) ; ...
       Initial.x.ang * ones(1,nray) ; ...
       Initial.y.pos * ones(1,nray) ; ...
       Initial.y.ang * ones(1,nray) ; ...
       Initial.zpos * ones(1,nray) ; ...
       Initial.Momentum * ones(1,nray) ] ;
  Q = Initial.Q * ones(1,nray) / nray ;
%
% loop over bunches
%
  for count = 1:Initial.NBunch
%
% assign x and Q to the bunch
%
      beamout.Bunch(count).x = x ;
      beamout.Bunch(count).Q = Q ;
%
% assign a Gaussian-distributed z & P, uncorrelated
%
      z = zeros(1,nray) ; P = zeros(1,nray) ;
%
% unfortunately, I think I'm obligated to do this the hard way
%
      zran = nsigz + 1 ; 
      for count2 = 1:nray
        zran = nsigz + 1 ; 
        while (abs(zran) > nsigz) 
          zran = randn(1) ;
        end
        z(count2) = zran ;
      end
      z = z * Initial.sigz ;
      Pcorrel = z * Initial.PZCorrel ;
      
      for count2 = 1:nray
        pran = nsigP + 1 ; 
        while (abs(pran) > nsigP) 
          pran = randn(1) ;
        end
        P(count2) = pran ;
      end
      P = P * Initial.SigPUncorrel  ;
      
      beamout.Bunch(count).x(5,:) = beamout.Bunch(count).x(5,:) + z ;
      beamout.Bunch(count).x(6,:) = beamout.Bunch(count).x(6,:) + P ...
                                  + Pcorrel ;
                              
  end
beamout = CheckBeamMomenta(beamout) ;
end

