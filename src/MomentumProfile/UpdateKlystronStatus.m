function UpdateKlystronStatus( klist )
%
% UPDATEKLYSTRONSTATUS Update status information in the KLYSTRON data
% structure.
%
%   UpdateKlystronStatus( klist ) examines the status of all klystrons in 
%      KLYSTRON indexed by klist.  Any klystron with a status of TRIPPED is
%      changed to TRIPSTANDBY, and any klystron with a status of MAKEUP is 
%      changed to ON.  

%========================================================================

  global KLYSTRON ;

  for count = 1:length(klist)
      
      if (strcmp(KLYSTRON(count).Stat,'TRIPPED'))
          KLYSTRON(count).Stat = 'TRIPSTANDBY' ;
      end
      if (strcmp(KLYSTRON(count).Stat,'MAKEUP'))
          KLYSTRON(count).Stat = 'ON' ;
      end
      
  end