function stat = SetGirderByBlock( istart, iend, is_long )
%
% SETGIRDERBYBLOCK Assign elements to girders according to their alignment
% block assignments.
%
% stat = SetGirderBySPos( istart, iend, is_long ) sets all elements
%    from istart to iend on girders according to their existing assignment
%    to alignment blocks.  Calling argument is_long indicates whether the
%    girder is supported at its ends (is_long==1) or at its center
%    (is_long==0).  Return argument stat is a Lucretia status and message
%    cell array (type help LucretiaStatus for more information).
%
% Return status value:  +1 for successful completion, 0 if an element is
% detected during assignment which already has a conflicting girder
% assignment.
%
% See also SetElementBlocks, AssignToGirder, SetGirderBySPos.
%

%==========================================================================

  global BEAMLINE GIRDER
  
  stat = InitializeMessageStack( ) ;
  girdno = length(GIRDER) + 1 ;
  
% begin looping
  
  count = istart ;
  while (count <=iend)
    if (isfield(BEAMLINE{count},'Block'))
      b = BEAMLINE{count}.Block ; b1 = min(b) ; b2 = max(b) ; 
      this_elist = [linspace(b1,b2,b2-b1+1)] ;
      count = b2+1 ;
    else
      this_elist = [count] ;
      count = count + 1 ;
    end
    e1 = min(this_elist) ;
    e2 = max(this_elist) ;
    S0 = BEAMLINE{e1}.S ;
    S1 = BEAMLINE{e2}.S ;
    if (isfield(BEAMLINE{e2},'L'))
      S1 = S1 + BEAMLINE{e2}.L ;
    end
    statcall = AssignToGirder( this_elist, girdno, is_long ) ;
    if (statcall{1} == 0)
      stat{1} = 0 ;
      stat = AddStackToStack(stat,statcall) ;
      return ;
    end
    if (statcall{1} == 1)
      girdno = girdno+1 ;
    end
  end

  