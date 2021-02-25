function stat = KlystronTrim( klyslist )

% KLYSTRONTRIM Set a klystron actual values to desired values
%
%   stat = KlystronTrim( Klystron_List ) sets the Ampl and Phase of each
%      KLYSTRON in the Klystron_List equal to the AmplSetPt and PhaseSetPt
%      of that device.  The step sizes are taken into account if they are
%      not zero, which can result in a KLYSTRON with a residual difference
%      in its desired and actual parameters.  Return argument stat is a
%      cell array, with stat{1} == 1 indicating success and stat{1} == 0
%      indicating error.  Error messages are transferred in stat{2...}.
%
% See also GirderMoverTrim, PSTrim.

%==========================================================================

  global KLYSTRON ;
  stat = InitializeMessageStack( ) ;
  if (max(klyslist) > length(KLYSTRON))
      stat{1} = 0 ;
      stat = AddMessageToStack(stat,...
          ['Out-of-range klystrons found in KlystronTrim']) ;
  end
  
% loop over klystrons

  for count = 1:length(klyslist) 
      klysno = klyslist(count) ;
      if (KLYSTRON(klysno).AmplStep == 0)
          KLYSTRON(klysno).Ampl = KLYSTRON(klysno).AmplSetPt ;
      else
          nstep = round( (KLYSTRON(klysno).AmplSetPt - ...
                          KLYSTRON(klysno).Ampl        ) / ...
                          KLYSTRON(klysno).AmplStep            ) ;
          KLYSTRON(klysno).Ampl = KLYSTRON(klysno).Ampl + ...
              nstep * KLYSTRON(klysno).AmplStep ;
      end
      if (KLYSTRON(klysno).PhaseStep == 0)
          KLYSTRON(klysno).Phase = KLYSTRON(klysno).PhaseSetPt ;
      else
          nstep = round( (KLYSTRON(klysno).PhaseSetPt - ...
                          KLYSTRON(klysno).Phase        ) / ...
                          KLYSTRON(klysno).PhaseStep            ) ;
          KLYSTRON(klysno).Phase= KLYSTRON(klysno).Phase + ...
              nstep * KLYSTRON(klysno).PhaseStep ;
      end
  end
      