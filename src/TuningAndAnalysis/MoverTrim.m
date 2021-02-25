function stat = MoverTrim( girdlist, varargin )

% MOVERTRIM Set girder mover actual values to desired values
%
%   stat = MoverTrim( Girder_List ) sets the MoverPos of each GIRDER in the
%      Girder_List equal to the MoverSetPt of that device. The step sizes
%      are taken into account if they are not zero, which can result in a
%      GIRDER with a residual difference in its desired and actual
%      parameters.  Return argument stat is a Lucretia status and message
%      cell array (type help LucretiaStatus for more information).
%
% Return status values: +1 if fully successful, -1 if members of
% Girder_List are past the end of the GIRDER array or do not have movers.
%
% See also KlystronTrim, PSTrim.

% MOD:
%      09-aug-2005, PT:
%         bugfix:  correct some typos in the stepsize/nostepsize
%         condition and in applying the correct # of steps to each
%         DOF.

%==========================================================================

  global GIRDER ;
  stat = InitializeMessageStack( ) ;
  if (max(girdlist) > length(GIRDER))
    stat = AddMessageToStack(stat,...
        'Out-of-range girders found in MoverTrim') ;
    stat{1} = -1 ;
  end
  
% loop over girders

  for count = 1:length(girdlist) 
      girdno = girdlist(count) ;
      if (girdno > length(GIRDER))
          continue ;
      end
      if (~isfield(GIRDER{girdno},'Mover'))
        stat = AddMessageToStack(stat,...
          'Girder without mover found in MoverTrim') ;
        stat{1} = -1 ;
        continue ;
      end
      for count2 = 1:length(GIRDER{girdno}.Mover)
        if length(GIRDER{girdno}.MoverStep)<count2
          GIRDER{girdno}.MoverStep(count2)=0;
        end
        if (GIRDER{girdno}.MoverStep(count2) == 0) 
           GIRDER{girdno}.MoverPos(count2) = ...
           GIRDER{girdno}.MoverSetPt(count2)     ;
        else
          nstep = round( (GIRDER{girdno}.MoverSetPt(count2) - ...
                          GIRDER{girdno}.MoverPos(count2)        ) / ...
                          GIRDER{girdno}.MoverStep(count2)            ) ;
          GIRDER{girdno}.MoverPos(count2) = GIRDER{girdno}.MoverPos(count2) + ...
              nstep * GIRDER{girdno}.MoverStep(count2) ;
        end
      end
  end
%  