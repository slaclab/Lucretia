function stat = PSTrim( pslist, varargin )

% PSTRIM Set power supply actual values to desired values
%
%   stat = PSTrim( PS_List ) sets the Ampl field of each PS in the PS_List
%      equal to the SetPt of that device.  The step sizes are taken into
%      account if they are not zero, which can result in a PS with a
%      residual difference in its desired and actual strength. Return
%      argument stat is a cell array, with stat{1} == 1 indicating success
%      and stat{1} == 0 indicating error.  Error messages are transferred
%      in stat{2...}.
%
% See also GirderMoverTrim, KlystronTrim.
%

%==========================================================================
% MODS:
% GW: March-11-2012
%   Allow PSElement and PS fields to make PSs hierarchical

global PS ;
stat = InitializeMessageStack( ) ;
if (max(pslist) > length(PS))
  stat{1} = 0 ;
  stat = AddMessageToStack(stat,...
    'Out-of-range power supplies found in PSTrim') ;
end


for count = 1:length(pslist)
  psno = pslist(count) ;
  if isfield(PS(psno),'PS') && ~isempty(PS(psno).PS) && PS(psno).PS % There is a master PS, add that strength to this PS
    mSet=PS(PS(psno).PS).Ampl;
  else
    mSet=0;
  end
  if (PS(psno).Step == 0)
    PS(psno).Ampl = mSet + PS(psno).SetPt ;
  else
    nstep = round( (PS(psno).SetPt - ...
      PS(psno).Ampl        ) / ...
      PS(psno).Step            ) ;
    PS(psno).Ampl = mSet + PS(psno).Ampl + ...
      nstep * PS(psno).Step ;
  end
  if isfield(PS(psno),'PSElement') && ~isempty(PS(psno).PSElement) && PS(psno).PSElement % If this PS controls other PSs loop over those
    stat=PSTrim(PS(psno).PSElement);
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      sprintf('Trim PS error (PS: %s)',stat{2})) ;
  end
end

