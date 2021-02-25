function stat = MovePhysicsVarsToKlystron( klyslist )

% MOVEPHYSICSVARSTOKLYSTRON exchange the dimensionless KLYSTRON Ampl and
%    dimensioned BEAMLINE Volt parameters
%
%    stat = MovePhysicsVarsToKlystron( klyslist ) exchanges each KLYSTRON's
%       Ampl parameter (nominally dimensionless) with it's structure's Volt
%       parameter, which nominally has dimensions of MV.  In the process
%       the KLYSTRON Ampl becomes the sum of all voltages of all RF units
%       supported by the KLYSTRON, and the BEAMLINE Volt parameters become
%       dimensionless scale factors (ie, if a KLYSTRON supports 10 equal
%       structures, then after the exchange each structure will have a Volt
%       of 0.1).  The AmplSetPt and AmplStep parameters are also rescaled,
%       and the mean structure phase is moved to the KLYSTRON phase, with
%       the differences remaining in the structures.  Each klystron in
%       klyslist gets exchanged in the manner described above. Return
%       variable stat is a Lucretia status and message stack (type help
%       LucretiaStatus for more information).
%
% Return status value == 1 for success, == 0 if klyslist includes klystrons
% which are out of range.

%==========================================================================

global BEAMLINE ;
global KLYSTRON ;
stat = InitializeMessageStack( ) ;  

% begin with range check 

for count = 1:length(klyslist)    
  klysno = klyslist(count) ;  

  if ( klysno > length(KLYSTRON) )
      stat = AddMessageToStack(stat,['Klystron # ',num2str(klysno),...
          ' out of range in MovePhysicsVarsToKlystron']) ;
      stat{1} = 0 ;
  end
end

if (stat{1} == 0)
    return ;
end

for count = 1:length(klyslist)    
  klysno = klyslist(count) ;  

% compute the total voltage and the mean phase

  V = 0 ; phi = 0 ;
  for count = 1:length(KLYSTRON(klysno).Element)
      elemno = KLYSTRON(klysno).Element(count) ;
      V = V + BEAMLINE{elemno}.Volt ;
      phi = phi + BEAMLINE{elemno}.Phase ;
  end
  
  phi = phi / length(KLYSTRON(klysno).Element) ;
  
% compute the scale factor.  If the klystron amplitude is zero, use unit
% scale factor

  if (KLYSTRON(klysno).Ampl ~=0)
    scale = V / KLYSTRON(klysno).Ampl ;
  else
    scale = 1 ;
  end
  
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
  
  % Ensure Volts add to 1 if KLYSTRON Ampl is 0
  for ikly=klyslist(:)'
    if KLYSTRON(ikly).Ampl==0
      Lrf=sum(arrayfun(@(x) BEAMLINE{x}.L,KLYSTRON(ikly).Element));
      for iele=KLYSTRON(ikly).Element
        BEAMLINE{iele}.Volt = BEAMLINE{iele}.L/Lrf ;
      end
    end
  end
  
end