function [stat,klist] = GetKlystronsInRange( istart, iend )
%
% GETKLYSTRONSINRANGE Make a list of klystrons which support RF structures
% within a specified range of elements.
%
%   [stat,klist] = GetKlystronsInRange( istart, iend ) finds all the members
%      of KLYSTRON which support RF structures between istart and iend in
%      BEAMLINE.  A list of their indices is returned in klist.  Return
%      argument stat is a status message cell array (type help
%      LucretiaStatus for more information).
%
%   Return status:  +1 if successful, -2 if klystrons were detected which
%   support devices both within and outside of the selected range.  In this
%   case, subsequent cells in stat contain messages on the problem
%   klystrons.

%==========================================================================

  global BEAMLINE ;
  global KLYSTRON ;
  
  stat = InitializeMessageStack( ) ;
  m = min(istart,iend) ; M = max(istart,iend) ;
  
  klist = [] ; iss = [] ;
  
  for count = 1:length(KLYSTRON)
      
      L = length(KLYSTRON(count).Element) ;
      a = KLYSTRON(count).Element >= m ; b = KLYSTRON(count).Element <= M ;
      if ( (sum(a)==0) | (sum(b)==0) )
          continue ;
      elseif ( (sum(a)==L) & (sum(b)==L) )
          klist = [klist count] ;
      else
          klist = [klist count] ;
          iss = [iss count] ;
      end
      
  end
  if (~isempty(iss)) 
      stat{1} = -2 ; 
      for count = 1:length(iss)
        stat = AddMessageToStack(stat,...
          ['Range problem detected:  KLYSTRON(',num2str(iss(count)),')'] ) ;
      end
  end