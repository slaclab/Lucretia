function stat = ConvertGirderToLong(igird)
%
% CONVERTGIRDERTOLONG Convert a girder from support at center to support at
% ends.
%
% stat = ConvertGirderToLong( igird ) converts girder igird from a short
%    girder which is supported at its center to a long girder which is
%    supported at its end.  Return argument stat is a Lucretia status and
%    message cell array (type help LucretiaStatus for more information).
%
% Return status values:  +1 if successful, 0 if girder igird does not
% exist or has no elements attached to it.
% 

%==========================================================================

  global GIRDER BEAMLINE
  stat = InitializeMessageStack( ) ;
  
  if (length(GIRDER)<igird)
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      ['Girder # ',num2str(igird),' does not exist in ConvertGirderToLong']) ;
    return ;
  end
  
  if (length(GIRDER{igird}.Element)<1)
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      ['Girder # ',num2str(igird),' has no elements in ConvertGirderToLong']) ;
    return ;
  end
  
% is the girder already long?

  if (length(GIRDER{igird}.S) == 2)
      return ;
  end
  
% otherwise get S positions

  e1 = GIRDER{igird}.Element(1) ; 
  e2 = GIRDER{igird}.Element(length(GIRDER{igird}.Element)) ;
  S0 = BEAMLINE{e1}.S ; S1 = BEAMLINE{e2}.S ;
  if (isfield(BEAMLINE{e2},'L'))
    S1 = S1 + BEAMLINE{e2}.L ;
  end
  GIRDER{igird}.S = [S0 S1] ;
