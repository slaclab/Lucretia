function stat = RenormalizeKlystron( klysno )

% RENORMALIZEKLYSTRON Scale and phase a klystron and its RF units such that
%    the klystron amplitude is returned to 1 and its phase to 0.
%
%    stat = RenormalizeKlystron( klysno ) scales the Ampl, AmplSetPt, and
%       AmplStep of a KLYSTRON such that Ampl == 1, and restores the Phase
%       to 0.  The inverse scale factor and the klystron phase are applied
%       to each BEAMLINE element supported by the klystron.  Return
%       variable stat is a cell array, with stat{1} == 1 if the exchange
%       occurred without error, == 0 if errors occurred, and stat{2...} are
%       text error messages.

%==========================================================================

  global BEAMLINE ;
  global KLYSTRON ;
  stat = InitializeMessageStack( ) ;
  
  if ( klysno > length(KLYSTRON) )
      stat = AddMessageToStack(stat,...
          ['Klystron # ',num2str(klysno),...
          ' out of range in MovePhysicsVarsToKlystron']) ;
      stat{1} = 0 ;
  end
  
% compute the scale factor and phase offset; if the klystron amplitude is
% zero, then we can't scale the RF but we can apply the phase
% transformation

  if (KLYSTRON(klysno).Ampl ~=0)
    scale = 1 / KLYSTRON(klysno).Ampl ;
  else
    scale = 1 ;
  end
  phi = -KLYSTRON(klysno).Phase ;
  
% apply the scale factor and the mean phase to the klystron

  KLYSTRON(klysno).Ampl = KLYSTRON(klysno).Ampl * scale ;
  KLYSTRON(klysno).AmplStep = KLYSTRON(klysno).AmplStep * scale ;
  KLYSTRON(klysno).AmplSetPt = KLYSTRON(klysno).AmplSetPt* scale ;
  KLYSTRON(klysno).Phase = KLYSTRON(klysno).Phase + phi ;
  KLYSTRON(klysno).PhaseSetPt = KLYSTRON(klysno).PhaseSetPt + phi ;
  
% now apply the reverse transformation on elements 

  for count = 1:length(KLYSTRON(klysno).Element)
      elemno = KLYSTRON(klysno).Element(count) ;
      BEAMLINE{elemno}.Volt = BEAMLINE{elemno}.Volt / scale ;
      BEAMLINE{elemno}.Phase = BEAMLINE{elemno}.Phase - phi ;
  end