function beamout = CreateBlankBeam( nbunch, nray, P0, bspace )
%
% CREATEBLANKBEAM Create a beam data structure for Lucretia.
%
% Beamout = CreateBlankBeam( nbunch, nray, P0, bspace ) creates
% a beam with nbunch bunches, nray rays (nray = scalar), 
% bunch spacing bspace, and all momenta set to P0.
%
% See also:  MakeBeamPZGrid, MakeBeamPZGauss, MakeBeam6DSparse, MakeBeam6DWeighted
%            MakeBeam6DGauss.
%
% Version date:  24-May-2007.

% MOD:
%      24-may-2007, PT:
%         check for bad transverse and total momentum before
%         returning the beam to the caller.

  beam.BunchInterval = bspace ;
  x = zeros(5,nray) ;
  x = [x ; P0 * ones(1,nray)] ;
  Q = ones(1,nray) * 1.60217653e-19;
  stop = zeros(1,nray) ;
  
  bunch.x = x ;
  bunch.Q = Q ;
  bunch.stop = stop ;
  
  beam.Bunch = [] ;
  for count = 1:nbunch
      beam.Bunch = [beam.Bunch ; bunch] ;
  end
  
  beamout = CheckBeamMomenta(beam) ;
  
  