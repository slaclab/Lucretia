function Bact = GetTrueStrength( indx , varargin )
%
% GETTRUESTRENGTH Compute the actual strength of a magnet
%
% Bact = GetTrueStrength( elem_index ) computes the B value of a magnet
%    taking into account both the magnet's B value and the amplitude of its
%    power supply, if any.
%
% Bact = GetTrueStrength( elem_index, 1 ) computes the strength of the
%    magnet on its power supply taking into account all slices of the magnet.
%
% See also:  GetTrueVoltage, GetTruePhase
%
% Version date:  23-May-2007.

% MOD: 
%      23-may-2007, PT:
%         support for multi-PS devices.
%      08-aug-2006, PT:
%         bugfix:  do the right thing if PS field exists but has a value
%         of zero.

%==========================================================================

  global BEAMLINE PS

  do_slices = 0 ;
  if (length(varargin) > 0)
    do_slices = varargin{1} ;
  end
  Bact = 0 ;
  if ( (isfield(BEAMLINE{indx},'Slices')) & (do_slices == 1) )
    for count = BEAMLINE{indx}.Slices
      Bact = Bact + BEAMLINE{count}.B ;
    end
  else
    Bact = BEAMLINE{indx}.B ;
  end
  if (isfield(BEAMLINE{indx},'PS'))
   psno = BEAMLINE{indx}.PS ;
   for count = 1:length(psno)
    if (psno(count)>0)
     Ampl = PS(psno(count)).Ampl ;
     Bact(count) = Bact(count) * Ampl ;
   end
   end
  end