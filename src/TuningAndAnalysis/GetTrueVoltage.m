function Bact = GetTrueVoltage( indx , varargin )
%
% GETTRUEVOLTAGE Compute the actual voltage of an RF structure
%
% Vact = GetTrueVoltage( elem_index ) computes the voltage of a structure
%    taking into account both the structure's Volt value and the amplitude
%    of its klystron, if any.
%
% Vact = GetTrueVoltage( elem_index, 1 ) computes the voltage of the
%    structure on its klystron taking into account all slices of the structure.
%
% See also:  GetTrueStrength, GetTruePhase
%

%==========================================================================

  global BEAMLINE KLYSTRON ;

  do_slices = 0 ;
  if (length(varargin) > 0)
    do_slices = varargin{1} ;
  end
  Bact = 0 ;
  if ( (isfield(BEAMLINE{indx},'Slices')) & (do_slices == 1) )
    for count = BEAMLINE{indx}.Slices
      Bact = Bact + BEAMLINE{count}.Volt ;
    end
  else
    Bact = BEAMLINE{indx}.Volt ;
  end
  if (isfield(BEAMLINE{indx},'Klystron')) && BEAMLINE{indx}.Klystron>0
    Ampl = KLYSTRON(BEAMLINE{indx}.Klystron).Ampl ;
    Bact = Bact * Ampl ;
  end