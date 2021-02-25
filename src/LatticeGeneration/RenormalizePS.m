function stat = RenormalizePS( psno )

% RENORMALIZEPS Rescale a power supply such that its amplitude == 1.0
%
%    stat = RenormalizePS( PSno ) rescales the Ampl, SetPt, and Step of a
%       power supply such that Ampl is returned to 1.0; the inverse of the
%       amplitude scale factor is applied to each magnet assigned to the
%       PS.  Return variable stat is a cell array, with stat{1} == 1 if the
%       exchange occurred without error, == 0 if errors occurred, and
%       stat{2...} are text error messages.

% MOD:
%        7-Sept-2011, GRW:
%           changed klyno->psno and added temp fix for bug where
%           BEAMLINE.PS list is correct but PS.Element list doesn't match
%           when BEAMLINE elements with same PS but different names
%        29-sep-2005, PT:
%           support for magnets with multiple power supplies.
%        28-sep-2005, PT:
%           improved handling of zero amplitude.

%==========================================================================

global BEAMLINE ;
global PS ;
stat = InitializeMessageStack( ) ;

% Is the desired PS in range?

if ( psno > length(PS) )
  stat = AddMessageToStack(stat,...
    ['PS # ',num2str(psno),...
    ' out of range in MovePhysicsVarsToPS']) ;
  stat{1} = 0 ;
end

% if the PS amplitude is zero, just go to its magnets and set their B
% values to 1
% (apply to all elements in same block with same PS)

if (PS(psno).Ampl == 0)
  for elemno = PS(psno).Element
    if ~elemno; continue; end
    if isfield(BEAMLINE{elemno},'Block') && ~isempty(BEAMLINE{elemno}.Block)
      for ibl=BEAMLINE{elemno}.Block(1):BEAMLINE{elemno}.Block(end)
        if isfield(BEAMLINE{ibl},'PS') && any(BEAMLINE{ibl}.PS==psno)
          if (length(BEAMLINE{ibl}.PS) == 1)
            BEAMLINE{ibl}.B = ones(1,length(BEAMLINE{ibl}.B)) ;
          else
            BEAMLINE{ibl}.B(BEAMLINE{ibl}.PS == psno) = 1 ;
          end
        end
      end
    else
      if (length(BEAMLINE{elemno}.PS) == 1)
        BEAMLINE{elemno}.B = ones(1,length(BEAMLINE{elemno}.B)) ;
      else
        BEAMLINE{elemno}.B(BEAMLINE{elemno}.PS == psno) = 1 ;
      end
    end
  end
  return ;
end

% compute the scale factor

scale = 1 / PS(psno).Ampl ;

% apply the scale factor to the PS

PS(psno).Ampl = PS(psno).Ampl * scale ;
PS(psno).Step = PS(psno).Step * scale ;
PS(psno).SetPt = PS(psno).SetPt * scale ;

% now apply the reverse transformation on elements
% (apply to all elements in same block with same PS)
doneibl=[];
for elemno = PS(psno).Element
  if ~elemno; continue; end
  if isfield(BEAMLINE{elemno},'Block') && ~isempty(BEAMLINE{elemno}.Block)
    for ibl=BEAMLINE{elemno}.Block(1):BEAMLINE{elemno}.Block(end)
      if isfield(BEAMLINE{ibl},'PS') && any(BEAMLINE{ibl}.PS==psno) && ~ismember(ibl,doneibl)
        if (length(BEAMLINE{elemno}.PS) == 1)
          BEAMLINE{ibl}.B = BEAMLINE{ibl}.B ./ scale ;
        else
          BEAMLINE{ibl}.B(BEAMLINE{ibl}.PS == psno) = BEAMLINE{ibl}.B(BEAMLINE{ibl}.PS == psno) ./ scale ;
        end
        doneibl(end+1)=ibl; %#ok<*AGROW>
      end
    end
  else
    if (length(BEAMLINE{elemno}.PS) == 1)
      BEAMLINE{elemno}.B = BEAMLINE{elemno}.B ./ scale ;
    else
      BEAMLINE{elemno}.B(BEAMLINE{elemno}.PS == psno) = BEAMLINE{elemno}.B(BEAMLINE{elemno}.PS == psno) ./ scale ;
    end
  end
end