function Bact = GetTruePhase( indx )
%
% GETTRUEVOLTAGE Compute the actual phase of an RF structure
%
% Pact = GetTruePhase( elem_index ) computes the phase of a structure
%    taking into account both the structure's Phase value and the phase
%    of its klystron, if any.
%
% See also:  GetTrueStrength, GetTrueVoltage
%

%==========================================================================

  global BEAMLINE KLYSTRON ;
  
  Bact = BEAMLINE{indx}.Phase ;
  if (isfield(BEAMLINE{indx},'Klystron')) && BEAMLINE{indx}.Klystron>0
    Ampl = KLYSTRON(BEAMLINE{indx}.Klystron).Phase ;
    Bact = Bact + Ampl ;
  end