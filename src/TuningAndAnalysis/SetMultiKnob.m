function stat = SetMultiKnob( knobname, Value, onlineCheck )
%
% SETMULTIKNOB Set the value of a multiknob.
%
% stat = SetMultiKnob( knobname, dValue, onlineCheck ) takes the _NAME_ of a
%    multiknob as a string and a value for the knob, and changes the values
%    of all parameters controlled by the knob by the desired amount.  At
%    the end of execution the value of the knob IN THE CALLER'S WORKSPACE
%    is updated with the new value.  
%
%    NOTE THAT THIS FUNCTION CHANGES THE VALUE OF THE KNOB IN THE CALLER'S
%    WORKSPACE!!!  
%
%    Return variable stat is a Lucretia status and message stack (type help
%    LucretiaStatus for more information).  Success is indicated by
%    stat{1}==1, failure by stat{1}==0.  If the knob failed to reach its
%    new set point, IncrementMultiKnob will attempt to restore all devices
%    to their status prior to execution of IncrementMultiKnob.
%
%    onlineCheck - only used in Floodland environment, otherwise ignore
%                - for Floodland =1 means change control system values
%                                =0 means simulation change only
%
% See also:  MakeMultiKnob, IncrementMultiKnob, RestoreMultiKnob.
%
% Version Date:  27-June-2006.
%

% MOD:
%      5-Nov-2008, GW: Floodland changes
%      27-jun-2006, PT:
%         use 16 digits in num2str conversion to minimize errors when
%         knobbing and restoring knobs.
%
%==========================================================================

% make a copy of the knob from the caller's workspace through clever and
% sneaky use of global variables

% Check passing name of knob, not knob itself
if ~ischar(knobname)
  error('Pass knob by name not knob itself')
end
  evalin('caller','global VIBRISSA') ;
  evalin('caller',['VIBRISSA = ',knobname,';']) ;
  global VIBRISSA ;
  knob = VIBRISSA ;
  clear global VIBRISSA ;
  
  dValue = Value - knob.Value ; 
  
  if exist('onlineCheck','var')
    stat = IncrementMultiKnob( 'knob', dValue, onlineCheck ) ;
  else
    stat = IncrementMultiKnob( 'knob', dValue ) ;
  end % if Floodland onlineCheck passed
  if ~iscell(stat{1}) && (stat{1}==1)
    evalin('caller',[knobname,'.Value=',num2str(Value,16),';']) ;
  else
    errstr=[];
    if iscell(stat{1})
      for icell=1:length(stat)
        errstr=[errstr stat{icell}{end}];
      end
      stat{1}=-1; stat{2}=errstr;
    end
  end
  
  