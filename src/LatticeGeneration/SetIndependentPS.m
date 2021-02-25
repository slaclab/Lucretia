function stat = SetIndependentPS( varargin )
%
% SETINDEPENDENTPS Assign all devices over a range to independent power
%    supplies.
%
%    stat = SetIndependentPS( istart, iend ) assigns each PS-valid device
%       in BEAMLINE within the range [istart,iend] to an independent power
%       supply (ie, no magnets are powered in series).  For devices which
%       are represented in slices, as indicated by a Slices field in the
%       BEAMLINE data structure, all slices will be powered by a single PS.
%       Return variable stat is a cell array, with stat{1} == 1 if the
%       exchange occurred without error, == 0 if errors occurred, and
%       stat{2...} are text error messages.
%
%    stat = SetIndependentPS( elemlist ) assigns each device in an element
%       list to an independent power supply, again putting all the slices
%       of a given element onto a single power supply.
%
% See also: SetElementSlices, AssignToPS
%
% Version date:  06-Mar-2008.

% MOD:
%      06-Mar-2008, PT:
%         bugfix:  only look at the length of the PS field for an element
%         when the AddFlag is set.
%      23-may-2007, PT:
%         support for XYCORs.  Support for solenoids.  Support for devices
%         which have more than 1 power supply.

%==========================================================================

  global BEAMLINE PS
  stat = InitializeMessageStack( ) ;  
  PS_ptr = length(PS) + 1 ;

% unpack arguments

  if (nargin == 1)
      elist = varargin{1} ;
  elseif (nargin == 2)
      elist = varargin{1}:varargin{2} ;
  else
      stat{1} = 0 ;
      stat = AddMessageToStack(stat, ...
          'SetIndependentPS:  Invalid arguments') ;
      return ;
  end
  
% loop over elements

%  for count = istart:iend
  for count = elist
      
      AddFlag = 0 ;
      
% check element validity

      if ( (strcmp(BEAMLINE{count}.Class,'QUAD'))     | ...
           (strcmp(BEAMLINE{count}.Class,'XYCOR'))    | ...
           (strcmp(BEAMLINE{count}.Class,'SBEN'))     | ...
           (strcmp(BEAMLINE{count}.Class,'XCOR'))     | ...
           (strcmp(BEAMLINE{count}.Class,'SEXT'))     | ...
           (strcmp(BEAMLINE{count}.Class,'OCTU'))     | ...
           (strcmp(BEAMLINE{count}.Class,'MULT'))     | ...
           (strcmp(BEAMLINE{count}.Class,'SOLENOID')) | ...
           (strcmp(BEAMLINE{count}.Class,'YCOR'))       )
           
% add this element to a PS if it has no slices, or if it has slices and it
% is the first element in its slice
           
          if (~isfield(BEAMLINE{count},'Slices') )
              AddFlag = 1 ;
              list = count ;
          elseif (BEAMLINE{count}.Slices(1) == count)
              AddFlag = 1 ;
              list = BEAMLINE{count}.Slices ;
          end
          
% what about the possibility of multiple power supplies for a device?
% Handle that now

          if ( (AddFlag == 1) & (isfield(BEAMLINE{count},'PS')) )
              AddFlag = length(BEAMLINE{count}.PS) ;
          end
          
          for psc = 1:AddFlag
            newstat = AssignToPS( list, PS_ptr, psc ) ;
            if (newstat{1} == 1)
                PS_ptr = PS_ptr + 1 ;
            else
                stat{1} = newstat{1} ;
                stat = AddStackToStack(stat,newstat) ;
            end
          end
          
      end
      
  end
              