function KtoB( istart, iend )
%
% KTOB convert MAD-style K values to Lucretia-style B values.
%
%   KtoB( istart, iend ) takes MAD-style K values (assumed to be stored in
%      the element B data fields) and converts to Lucretia's native B
%      format.  This permits a Lucretia BEAMLINE to be built from a MAD
%      deck, temporarily storing K values in the BEAMLINE B fields, and
%      then converting to B once the momentum profile has been computed.

%========================================================================

  global BEAMLINE ;

  brhofact = 1/0.299792458 ; % convert between GeV/c and T.m

  for count = min(istart,iend):max(istart,iend)
      
      switch BEAMLINE{count}.Class
          
          case{ 'XCOR' , 'YCOR' , 'SBEN' , 'RBEN', 'MULT' }
              
              BEAMLINE{count}.B = BEAMLINE{count}.B * ...
                                  BEAMLINE{count}.P * brhofact ;
          case{ 'QUAD' , 'SEXT' , 'OCTU' }
              BEAMLINE{count}.B = BEAMLINE{count}.B * ...
                                  BEAMLINE{count}.L * ...
                                  BEAMLINE{count}.P * brhofact ;
          case{ 'SBEN' }
              BEAMLINE{count}.B = BEAMLINE{count}.B * ...
                                  BEAMLINE{count}.P * brhofact ;
              if (length(BEAMLINE{count}.B) > 1)
                  BEAMLINE{count}.B(2) = BEAMLINE{count}.B(2) * ...
                                         BEAMLINE{count}.L ;
              end      
                 
      end
      
  end
              