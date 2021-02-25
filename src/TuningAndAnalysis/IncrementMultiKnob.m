function stat = IncrementMultiKnob( knobname, dValue, onlineCheck )
%
% INCREMENTMULTIKNOB Increment the value of a multiknob.
%
% stat = IncrementMultiKnob( knobname, dValue, onlineCheck ) takes the _NAME_ of a
%    multiknob as a string and a desired change in the knob's value, and
%    changes the values of all parameters controlled by the knob by the
%    desired amount.  At the end of execution the value of the knob IN THE
%    CALLER'S WORKSPACE is updated with the new value.
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
% See also:  MakeMultiKnob, SetMultiKnob, RestoreMultiKnob.
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

global KLYSTRON GIRDER PS %#ok<NUSED>
stat{1}=1; stat{2}=[];

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

% loop over channels

nchannel = length(knob.Channel) ;
ChannelV1 = zeros(nchannel,1) ;
ChannelV2 = zeros(nchannel,1) ;
for count = 1:nchannel
  
  % get the original and new channel values
  
  eval(['ChannelV1(count) = ',knob.Channel(count).Parameter,';']) ;
  ChannelV2(count) = ChannelV1(count) + ...
    knob.Channel(count).Coefficient * dValue ;
  
end

% loop over channels again and attempt to trim devices

for count = 1:nchannel
  
  eval([knob.Channel(count).Parameter,'=',...
    num2str(ChannelV2(count),16),';']) ;
  
  % execute an appropriate device trim
  
  if ( strncmp(knob.Channel(count).Parameter(1:2),'PS',2) )
    if exist('onlineCheck','var')
      stat2 = PSTrim(knob.Channel(count).Unit, onlineCheck) ;
      if count<nchannel
        pause(1);
      end
    else
      stat2 = PSTrim(knob.Channel(count).Unit) ;
    end % if onlineCheck passed (using Floodland)
  elseif ( strncmp(knob.Channel(count).Parameter(1:6),'GIRDER',6) )
    if exist('onlineCheck','var')
      stat2 = MoverTrim(knob.Channel(count).Unit, onlineCheck) ;
      if count<nchannel
        pause(1);
      end
    else
      stat2 = MoverTrim(knob.Channel(count).Unit) ;
    end % if onlineCheck passed (using Floodland)
  elseif ( strncmp(knob.Channel(count).Parameter(1:8),'KLYSTRON',8) )
    stat2 = KlystronTrim(knob.Channel(count).Unit) ;
  end
  
  if (stat2{1} ~=1)
    stat{1}=-1;
    stat{2}=[stat{2} 'Trim failure for channel: ',num2str(count),' '];
  end
  
end

% if success, set the new knob value in the caller workspace and exit

if ~iscell(stat{1}) || (stat{1} == 1)
  
  evalin('caller',[knobname,'.Value=',knobname,'.Value+',...
    num2str(dValue,16),';'] ) ;
  return ;
  
end

% otherwise try to clean up

count2 = count ;
for count = 1:count2-1
  
  eval([knob.Channel(count).Parameter,'=',...
    num2str(ChannelV1(count),16),';']) ;
  
  % execute an appropriate device trim
  
  if ( strncmp(knob.Channel(count).Parameter(1:2),'PS',2) )
    if exist('onlineCheck','var')
      stat3 = PSTrim(knob.Channel(count).Unit, onlineCheck) ;
    else
      stat3 = PSTrim(knob.Channel(count).Unit) ;
    end % if onlineCheck passed (using Floodland)
  elseif ( strncmp(knob.Channel(count).Parameter(1:2),'GIRDER',6) )
    if exist('onlineCheck','var')
      stat3 = MoverTrim(knob.Channel(count).Unit, onlineCheck) ;
    else
      stat3 = MoverTrim(knob.Channel(count).Unit) ;
    end % if onlineCheck passed (using Floodland)
  elseif ( strncmp(knob.Channel(count).Parameter(1:2),'KLYSTRON',8) )
    stat3 = KlystronTrim(knob.Channel(count).Unit) ;
  end
  
  if (stat3{1} ~=1)
    stat{2}=[stat{2} 'Failed to untrim Channel: ',num2str(count), 'return by hand'];
  end
  
end
