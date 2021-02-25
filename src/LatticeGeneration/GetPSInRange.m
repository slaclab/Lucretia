function [stat,plist] = GetPSInRange( istart, iend )
%
% GETPSINRANGE Make a list of power supplieswhich support magnets
%    within a specified range of elements.
%
%   [stat,plist] = GetPSInRange( istart, iend ) finds all the members
%      of PS which support magnets between istart and iend in BEAMLINE.  A
%      list of their indices is returned in plist.  Return argument stat is
%      a status message cell array (type help LucretiaStatus for more
%      information).
%
%   Return status:  +1 if successful, -1 if power supplies were detected
%   which support devices both within and outside of the selected range.
%   In this case, subsequent cells in stat contain messages on the problem
%   power supplies.

%==========================================================================

  global BEAMLINE ;
  global PS ;

  stat = InitializeMessageStack( ) ;  
  m = min(istart,iend) ; M = max(istart,iend) ;
  
  plist = [] ; iss = [] ;
  
  for count = 1:length(PS)
      
      L = length(PS(count).Element) ;
      a = PS(count).Element >= m ; b = PS(count).Element <= M ;
      if ( (sum(a)==0) | (sum(b)==0) )
          continue ;
      elseif ( (sum(a)==L) & (sum(b)==L) )
          plist = [plist count] ;
      else
          plist = [plist count] ;
          iss = [iss count] ;
      end
      
  end
  if (~isempty(iss)) 
      stat{1} = -1 ; 
      for count = 1:length(iss)
        stat = AddMessageToStack(stat,...
          ['Range problem detected:  PS',num2str(iss(count)),')'] ) ;
      end
  end