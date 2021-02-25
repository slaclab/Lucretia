function stat = AddBPMToLcav( elemno, numbpm )
%
% ADDBPMTOLCAV Add one or more HOM-BPMs to an RF structure
%
% stat = AddBPMToLcav( element_number, num_bpm ) adds one or more HOM BPMs
%    to the data structure for an RF accelerating structure ("LCAV") and
%    deletes any existing BPMs on that structure.  Argument element_number
%    is the index into the BEAMLINE array of the RF structure, num_bpm is
%    the number of HOM-BPMs to be added.  Return variable stat is a
%    Lucretia status and message cell array (type help LucretiaStatus for
%    more information).
%
% Return status:  +1 for success, 0 if BEAMLINE{element_number} is not an
% RF structure.
%

%==========================================================================

stat = InitializeMessageStack() ;
global BEAMLINE ;
if (~strcmp(BEAMLINE{elemno}.Class,'LCAV'))
  stat{1} = 0 ;
  stat = AddMessageToStack(stat,...
      ['Element # ',num2str(elemno),' is not an LCAV']) ;
  return ;
end

BEAMLINE{elemno}.NBPM = numbpm ;
BEAMLINE{elemno}.BPMOffset = zeros(2,numbpm) ;
