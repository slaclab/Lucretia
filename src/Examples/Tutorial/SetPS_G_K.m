function stat = SetPS_G_K
%
% SetPS_G_K -- set the power supplies, klystrons, and girders for the
% tutorial linac lattice.
%
% Version date:  12-mar-2008

% Modification history:
%
%==========================================================================

  global BEAMLINE PS KLYSTRON GIRDER ;

% Power supplies:  power the quads and correctors independently, and the
% final bend chicane has 1 power supply and the bends are powered in series

  blist = findcells(BEAMLINE,'Class','SBEN') ;
  stat = SetIndependentPS(1,blist(1)-1) ;
  if (stat{1} == 1)
      stat = AddMessageToStack(stat,'SetIndependentPS:  OK') ;
  end
  
  stat1 = AssignToPS(blist,length(PS)+1) ;
  stat = AddStackToMasterStack(stat,stat1,'AssignToPS') ;
  
% klystrons: assign 1 klystron per 16 cavities -- different from ILC or any
% other proposed facility, but it fits nicely into the design of this
% lattice

  stat1 = SetKlystrons(1,length(BEAMLINE),16) ;
  stat = AddStackToMasterStack(stat,stat1,'SetKlystrons') ;
  
% now for the girders -- we want to assign 8 cavities plus one quad package
% (corrector, quad, BPM) to each girder.  Start by figuring out how many
% girders there are (it's equal to the number of xycors)

  xylist = findcells(BEAMLINE,'Class','XYCOR') ;
  NGirdersNeeded = length(xylist) ;
  
% get a list of lcavs and BPMs -- each girder starts with an lcav and ends
% with a BPM

  lcavlist = findcells(BEAMLINE,'Class','LCAV') ;
  monilist = findcells(BEAMLINE,'Class','MONI') ;
  
% loop over girders and assign elements.  Girder M will end with the Mth
% BPM, but will start with the 8*(M-1)+1'th cavity.  Each girder is long
% (ie supported at its ends and not its center), so the last argument in
% AssignToGirder should be 1, for a long girder, not 0, which would be for
% a short girder.

  for gcount = 1:NGirdersNeeded
      gstart = lcavlist(8*(gcount-1)+1) ;
      gstop  = monilist(gcount) ;
      glist = gstart:gstop ;
      stat1 = AssignToGirder(glist,-1,1) ;
      stat = AddStackToMasterStack(stat,stat1,...
          ['AssignToGirder, linac girder ',num2str(gcount)]) ;
  end
  
% now assign the chicane bends and the chicane BPM to their own girder

  sblist = findcells(BEAMLINE,'Class','SBEN') ;
  blist = findcells(BEAMLINE,'Class','MONI',sblist(1),sblist(end)) ;
  stat1 = AssignToGirder([blist sblist],length(GIRDER)+1,1) ;
  stat = AddStackToMasterStack(stat,stat1,'AssignToGirder, bend girder') ;
  
% assign the wire scanner to its own girder -- it's separately supported
% from the floor, as wire scanners often are.  Unlike the other girders
% assigned so far, this one is short, so the last argument is a 0 and not a
% 1.

  wlist = findcells(BEAMLINE,'Class','WIRE') ;
  stat1 = AssignToGirder(wlist,length(GIRDER)+1,0) ;
  stat = AddStackToMasterStack(stat,stat1,'AssignToGirder, BPM girder') ;
