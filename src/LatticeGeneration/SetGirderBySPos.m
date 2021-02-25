function stat = SetGirderBySPos( istart, iend, ds, is_long )
%
% SETGIRDERBYSPOS Assign elements to girders according to their S
% positions.
%
% stat = SetGirderBySPos( istart, iend, ds, is_long ) sets all elements
%    from istart to iend on girders of specified length ds.  Calling
%    argument is_long indicates whether the girder is supported at its ends
%    (is_long==1) or at its center (is_long==0).  Elements which are in a
%    defined alignment block will all be assigned to a common girder.
%    Because of blocks, and because constraining girders to an exact length
%    of ds might cause a girder to terminate "in the middle" of an element,
%    actual girder lengths will only be approximately equal to ds.  Return
%    argument stat is a Lucretia status and message cell array (type help
%    LucretiaStatus for more information).
%
% Return status value:  +1 for successful completion, 0 if an element is
% detected during assignment which already has a conflicting girder
% assignment.
%
% See also SetElementBlocks, AssignToGirder.
%

%==========================================================================

  global BEAMLINE GIRDER
  
  stat = InitializeMessageStack( ) ;
  girdno = length(GIRDER) + 1 ;
  
% begin looping
  
  this_elist = [] ; 
  count = istart ;
  while (count <=iend)
    if (isempty(this_elist))
      this_elist = count ;
      S0 = BEAMLINE{this_elist}.S ;
    end
    if (isfield(BEAMLINE{count},'Block'))
      b = BEAMLINE{count}.Block ; b1 = min(b) ; b2 = max(b) ; 
      this_elist = [this_elist linspace(b1,b2,b2-b1+1)] ;
      count = b2+1 ;
    else
      this_elist = [this_elist count] ;
      count = count + 1 ;
    end
    e2 = max(this_elist) ;
    S1 = BEAMLINE{e2}.S ;
    if (isfield(BEAMLINE{e2},'L'))
      S1 = S1 + BEAMLINE{e2}.L ;
    end
    if (S1 >= S0 + ds)
      statcall = AssignToGirder( this_elist, girdno, is_long ) ;
      if (statcall{1} == 0)
        stat{1} = 0 ;
        stat = AddStackToStack(stat,statcall) ;
        return ;
      end
      this_elist = [] ;
      if (statcall{1} == 1)
        girdno = girdno+1 ;
      end
    end
  end
  