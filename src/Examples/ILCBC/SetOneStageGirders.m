function stat = SetOneStageGirders( )

  global GIRDER BEAMLINE ;
  stat = InitializeMessageStack( ) ;

  i0 = 1 ; i1 = findcells(BEAMLINE,'Name','BC0END') ;
           i2 = findcells(BEAMLINE,'Name','BC1RFEND') ;
           i3 = findcells(BEAMLINE,'Name','ELIN1MATCHEND') ;
           i4 = findcells(BEAMLINE,'Name','ELIN3END') ;
           
% begin with the initial matching region

  stat1 = SetGirderByBlock(i0,i1,0) ;
  stat{1} = stat1{1} ;
  stat = AddStackToStack(stat,stat1) ;
  
% now for the BC1 RF: find the D_END drifts at the ends of
% the cryomodules

  dlist = findcells(BEAMLINE,'Name','D_END',i1,i2) ;

% loop over modules and girderize

  for ist = 1:2:length(dlist)
    ngird = length(GIRDER) ;
    istart = dlist(ist) ; iend = dlist(ist+1) ;
    stat1 = AssignToGirder( istart:iend, ngird+1, 1 ) ;
    stat{1} = [stat{1} stat1{1}] ;
    stat = AddStackToStack(stat,stat1) ;
  end
    
% set the wiggler and the various matching quads on independent girders

  stat1 = SetWigglerGirders(i2,i3) ;
  stat{1} = [stat{1} stat1{1}] ;
  stat = AddStackToStack(stat,stat1) ;
  
% now the main linac including its matching modules

  dlist = findcells(BEAMLINE,'Name','D_END',i3,i4) ;

% loop over modules and girderize

  for ist = 1:2:length(dlist)
    ngird = length(GIRDER) ;
    istart = dlist(ist) ; iend = dlist(ist+1) ;
    stat1 = AssignToGirder( istart:iend, ngird+1, 1 ) ;
    stat{1} = [stat{1} stat1{1}] ;
    stat = AddStackToStack(stat,stat1) ;
  end
    
  stat{1} = min(stat{1}) ;