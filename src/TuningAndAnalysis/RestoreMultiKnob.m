function stat = RestoreMultiKnob( knobname, onlinecheck )
%
% RESTOREMULTIKNOB Restore the value of a multiknob to zero.
%
% stat = RestoreMultiKnob( knobname ) takes the _NAME_ of a multiknob as a 
%    string and sets that multiknob's value to zero. At the end of
%    execution the knob IN THE CALLER'S WORKSPACE is updated with the new
%    value of zero.  
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
% See also:  MakeMultiKnob, IncrementMultiKnob, SetMultiKnob.
%
% Version Date:  27-June-2006.
%

% MOD:
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
  
  dValue =  -knob.Value ; 
  
  if exist('onlinecheck','var') && onlinecheck
    stat = IncrementMultiKnob( 'knob', dValue, 1 ) ;
  else
    stat = IncrementMultiKnob( 'knob', dValue ) ;
  end
  if (stat{1}==1)
    evalin('caller',[knobname,'.Value=0;']) ;
  end
  
