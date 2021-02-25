function stat = SetStandbyKlystron( BunchCharge ) ;
%
% SetStandbyKlystron -- reduce total gradient, put the last klystron in the
% linac into standby state, rescale the optics.
%
% Version date:  13-mar-2008.

% Revision history:
%
%==========================================================================

  global BEAMLINE KLYSTRON ;
  
% find the last cavity driven by the next-to-last klystron

  Nklys = length(KLYSTRON) ;
  cav = KLYSTRON(Nklys-1).Element ; lastcav = cav(end) ;
  
% find the momentum after the last cavity and at injection

  Pf = BEAMLINE{lastcav+1}.P ;
  P0 = BEAMLINE{1}.P ;
  
% round down to the nearest 1/2 GeV

  NHalfGev = floor(2*Pf) ; Pfdes = NHalfGev/2 ;
  
% Set the last klystron to be in STANDBY state, and scale the linac to the
% desired final momentum.  

  KLYSTRON(Nklys).Stat = 'STANDBY' ;
  stat = SetDesignMomentumProfile(1,length(BEAMLINE),BunchCharge,P0,Pfdes) ;
  if (stat{1} == 1)
    stat = AddMessageToStack(stat,'SetDesignMomentumProfile:  OK') ;
  end
 
% finally, tune the amplitude of the last klystron, which is on STANDBY
% status, to match that of the others.  This is so that, if there's a need
% to swap klystrons, the STANDBY tube has the same energy gain as the
% current set of ON tubes.

  KLYSTRON(Nklys).AmplSetPt = KLYSTRON(Nklys-1).AmplSetPt ;
  stat1 = KlystronTrim(Nklys) ;
  stat = AddStackToMasterStack(stat,stat1,'KlystronTrim') ;
  