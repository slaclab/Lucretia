function stat = SetKlystrons( istart, iend, inum, varargin )
%
% SETKLYSTRONS Assign all RF devices over a range to klystrons.
%
%    stat = SetKlystrons( istart, iend, inum ) assigns each RF device
%       in BEAMLINE within the range [istart,iend] to a klystron.  Argument
%       inum indicates the number of RF devices to be assigned to each 
%       KLYSTRON.  Both LCAVs and TCAVs will be assigned to klystrons, and
%       both types of device can and will be assigned to a single klystron.
%       For devices which are represented in slices, as indicated by a
%       Slices field in the BEAMLINE data structure, all slices will be
%       treated as a single device for the purposes of assignment. Return
%       variable stat is a cell array, with stat{1} == 1 if the exchange
%       occurred without error, == 0 if errors occurred, and stat{2...} are
%       text error messages.
%
%    stat = SetKlystrons( istart, iend, inum, mode ) assigns only devices
%       of a specified type to klystrons.  If mode==0, LCAVs are assigned
%       to klystrons.  If mode==1, TCAVs are assigned to klystrons.
%
% See also: SetElementSlices, AssignToKlystron

% MOD:
%       30-sep-2005, PT:
%           support for TCAVs.

%==========================================================================

  if (nargin==3)
    mode = 2 ; % mixed assignment mode 
  end
  if (nargin>4)
    error('Invalid arguments detected in SetKlystrons.m') ;
  end
  if (nargin==4)
    switch varargin{1}
        case 0
            mode = 0 ; % LCAVs only
        case 1
            mode = 1 ; % TCAVs only
        otherwise
            error('Invalid arguments detected in SetKlystrons.m') ;
    end
  end
  global BEAMLINE KLYSTRON
  stat = InitializeMessageStack( ) ;
  
  PS_ptr = length(KLYSTRON) + 1 ;
  list = [] ;
  iklys = 1 ;
  
% loop over elements

  for count = istart:iend
      
% check element validity

      if ( ( (strcmp(BEAMLINE{count}.Class,'LCAV')) & ( (mode==0)|(mode==2) ) ) | ...       
           ( (strcmp(BEAMLINE{count}.Class,'TCAV')) & ( (mode==1)|(mode==2) ) )       )      
           
% add this element to a klystron if it has no slices, or if it has slices and it
% is the first element in its slice
           
          if (~isfield(BEAMLINE{count},'Slices'))
              list = [list count] ;
              iklys = iklys + 1 ;
          elseif (BEAMLINE{count}.Slices(1) == count)
              list = [list BEAMLINE{count}.Slices] ;
              iklys = iklys + 1 ;
          end
          
          if (iklys == inum + 1) 
            newstat = AssignToKlystron( list, PS_ptr ) ;
            if (newstat{1} == 1)
                PS_ptr = PS_ptr + 1 ;
                if (PS_ptr == 337)
                    disp(' ') ;
                end
                iklys = 1 ;
                list = [] ;
            else
                stat{1} = newstat{1} ;
                stat = AddStackToStack(stat,newstat) ;
            end
          end
          
      end
      
  end
  
% If the last klystron has too few RF units (ie < iklys), then the above
% loop will never assign it.  Handle that now.

  if (~isempty(list)) 
    newstat = AssignToKlystron( list, PS_ptr ) ;
    if (newstat{1} ~= 1)
        stat{1} = newstat{1} ;
        stat = AddStackToStack(stat,newstat) ;
    end
  end

              