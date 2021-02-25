function [stat,Initial,nuxdeg,nuydeg] = GenerateTutorialInitial( charge, nbunch, dt )
%
% GenerateTutorialInitial -- generate an initial conditions data structure
% for the Lucretia tutorial.
%
% [stat,Initial,nux,nuy] = GenerateTutorialInitial( charge , nbunch , dt )
% fills up an initial conditions data structure and returns it to the main
% caller. Arguments are the charge per bunch, number of bunches, and bunch
% spacing in time (seconds).  The function also accesses the BEAMLINE
% global data structure to find the appropriate matched Twiss conditions
% for the injection points.  The phase advance per cell, in degrees, is
% returned in nux and nuy arguments, and overall status is returned in
% stat.
%
% Version date:  11-Mar-2008.

% Modification history:
%
%==========================================================================

  global BEAMLINE 
  stat = InitializeMessageStack() ;
  
% get a blank Initial condition structure

  Initial = InitCondStruc() ;
  
% fill with data

  Initial.Momentum = 5 ;  Initial.SigPUncorrel = 0.01 * 5 ;
  Initial.Q = charge ; Initial.NBunch = nbunch ;
  Initial.BunchInterval = dt ;
  Initial.x.NEmit = 2e-6 ; Initial.y.NEmit = 2e-8 ;
  Initial.sigz = 1e-3 ;
  
% find the first FODO cell and compute its stable Twiss parameters.  If the
% cell is unstable, set bad status on exit.

  istart = 1 ; dlist = findcells(BEAMLINE,'Name','Drift30cm') ; istop = dlist(2) ;
  
  try
      Initial.x.Twiss = ReturnMatchedTwiss(istart,istop,1) ;
  catch
      stat{1} = 0 ;
      stat{2} = 'Unstable x plane, unable to compute Twiss!' ;
      return ;
  end
  nuxdeg = Initial.x.Twiss.nu * 360 ;
  Initial.x.Twiss.nu = 0 ;
  try
      Initial.y.Twiss = ReturnMatchedTwiss(istart,istop,2) ;
  catch
      stat{1} = 0 ;
      stat{2} = 'Unstable y plane, unable to compute Twiss!' ;
      return ;
  end
  nuydeg = Initial.y.Twiss.nu * 360 ;
  Initial.y.Twiss.nu = 0 ;
  
  