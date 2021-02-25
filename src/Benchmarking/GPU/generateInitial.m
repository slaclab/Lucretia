function [Initial,nuxdeg,nuydeg] = generateInitial( charge, nbunch, dt )
%
  global BEAMLINE
  
% get a blank Initial condition structure

  Initial = InitCondStruc() ;
  
% fill with data

  Initial.Momentum = 5 ;  Initial.SigPUncorrel = 0.01 * 5 ;
  Initial.Q = charge ; Initial.NBunch = nbunch ;
  Initial.BunchInterval = dt ;
  Initial.x.NEmit = 2e-6 ; Initial.y.NEmit = 2e-8 ;
  Initial.sigz = 1e-3 ;
  
% find the first FODO cell and compute its stable Twiss parameters. 

  istart = 1 ; dlist = findcells(BEAMLINE,'Name','Drift30cm2') ; istop = dlist(2) ;
  
  Initial.x.Twiss = ReturnMatchedTwiss(istart,istop,1) ;
  nuxdeg = Initial.x.Twiss.nu * 360 ;
  Initial.x.Twiss.nu = 0 ;
  Initial.y.Twiss = ReturnMatchedTwiss(istart,istop,2) ;
  nuydeg = Initial.y.Twiss.nu * 360 ;
  Initial.y.Twiss.nu = 0 ;
  
% Correlated energy spread to facilitate bunch length compression
Initial.PZCorrel = 0 ;