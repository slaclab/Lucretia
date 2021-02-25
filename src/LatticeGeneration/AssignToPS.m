function stat = AssignToPS( elemlist, psno, varargin )

% ASSIGNTOPS Assign elements to a power supply.
%
%   stat = AssignToPS( element_list, PS_number ) assigns the
%      BEAMLINE elements in the element_list to the power supply indexed by
%      PS_number.  The elements are added to any existing ones which are
%      already assigned to the power supply.  If PS_number is greater than
%      the present length of the PS array, it will be extended; if the PS
%      array is missing, it will be created; if PS_number == 0, all PS
%      assignments for devices in element_list will be cleared. If all
%      devices on a PS are at zero field, the device B values will be set
%      to 1 and the PS Ampl and SetPt will be set to zero.  Return argument
%      stat is a cell array, with stat{1} the status and stat{2...} text
%      messages from the function. The value of stat{1} == 1 for error-free
%      execution, or == 0 if error occurs (invalid device in the element
%      list and/or element in the element list which already has a power
%      supply).  In the event of error, no assignments will be performed.
%
%   stat = AssignToPS( element_list, PS_number, PS_index ) allows the user 
%      to assign one power supply on a multi-PS element to the desired
%      device in the PS data structure.
%
%   Note:  A gradient bend with B == [0 0] cannot be assigned to a single
%     power supply which excites both bending and focusing fields, but can
%     be assigned to separate power supplies for the two fields.  This is
%     because, in the case of B == [0 0], there is no way for the
%     simulation to determine the ratio of bending to focusing to apply
%     when a single power supply is excited to a nonzero setpoint.
%
% Version date: 22-May-2007.

% MOD:
%      22-May-2007, PT:
%         support for XYCOR (which has 2 power supplies).  Improvements in
%         the general machinery for multiple power supplies driving a
%         device.
%      12-Jun-2006, PT:
%         bugfix:  element list is forced into column-vector form, which
%         prevents 2 element lists from properly concatenating.
%      09-Mar-2006, PT:
%         support for solenoids.
%      28-sep-2005, PT:
%         support for multi-PS devices.

%==========================================================================

% is PS around?  If not, create it

  stat = InitializeMessageStack( ) ;
  global BEAMLINE ;
  global PS ;
  
% default is to assign the first (and usually only) power supply for an
% element to the desired PS, but user can specify differently

  PS_index = 1 ; MultiPS = 0 ;
  if ( nargin == 3)
    PS_index = varargin{1} ;
    MultiPS = 1 ;
  end
  
% make sure that the new devices are not yet assigned to power supplies,
% and are valid PS devices

  Btot = 0 ;
  elemlist = elemlist(:) ;
  oldpslist = [] ;
  for count = 1:length(elemlist)
      elemno = elemlist(count) ;
      
% general:  check whether the device is permitted to have a PS  

      if (  (~strcmp(BEAMLINE{elemno}.Class,'XCOR'))     && ...
            (~strcmp(BEAMLINE{elemno}.Class,'XYCOR'))    && ...
            (~strcmp(BEAMLINE{elemno}.Class,'YCOR'))     && ...
            (~strcmp(BEAMLINE{elemno}.Class,'QUAD'))     && ...
            (~strcmp(BEAMLINE{elemno}.Class,'SEXT'))     && ...
            (~strcmp(BEAMLINE{elemno}.Class,'OCTU'))     && ...
            (~strcmp(BEAMLINE{elemno}.Class,'MULT'))     && ...
            (~strcmp(BEAMLINE{elemno}.Class,'SOLENOID')) && ...            
            (~strcmp(BEAMLINE{elemno}.Class,'SBEN'))       )
         stat{1} = 0 ;
         stat = AddMessageToStack(stat,...
             ['Invalid element # ',num2str(elemno),...
             ' detected in AssignToPS input']) ;
         
% if we are trying to assign an extra PS to a device make sure it's
% permitted

      elseif ( (PS_index > 1) && ...
               (~strcmp(BEAMLINE{elemno}.Class,'SBEN')) && ...
               (~strcmp(BEAMLINE{elemno}.Class,'XYCOR'))  ...
             )
         stat{1} = 0 ;
         stat = AddMessageToStack(stat,...
             ['Element # ',num2str(elemno),...
             ' cannot have > 1 PS in AssignToPS input']) ;
      elseif ( (PS_index > 1) && ...
               (length(BEAMLINE{elemno}.B) < 2)   )
         stat{1} = 0 ;
         stat = AddMessageToStack(stat,...
             ['Element # ',num2str(elemno),...
             ' cannot have > 1 PS in AssignToPS input']) ;
         
% at the moment only 2 PS are allowed on any device    

      elseif ( PS_index > 2 )
         stat{1} = 0 ;
         stat = AddMessageToStack(stat,...
             ['Element # ',num2str(elemno),...
             ' cannot have > 2 PS in AssignToPS input']) ;
      else    
          
% if the PS field is empty or entirely missing, add it now          
          
         if (~isfield(BEAMLINE{elemno},'PS'))
            BEAMLINE{elemno}.PS = zeros(1,PS_index) ;
         end
         if (isempty(BEAMLINE{elemno}.PS))
            BEAMLINE{elemno}.PS = zeros(1,PS_index) ;
         end
             
% look for conflicting PS assignment           

         if ( (~isempty(BEAMLINE{elemno}.PS))            && ...
              (length(BEAMLINE{elemno}.PS)>= PS_index)   && ...
              (BEAMLINE{elemno}.PS(PS_index) ~= 0)       && ...
              (BEAMLINE{elemno}.PS(PS_index) ~= psno) && ...
              (psno ~= 0)                         )
             stat{1} = 0 ;
             stat = AddMessageToStack(stat,...
                ['Element # ',num2str(elemno), ...
                 ' already has conflicting PS assignment in AssignToPS']) ;
         end
         if (psno == 0)
             oldpslist = [oldpslist BEAMLINE{elemno}.PS(PS_index)] ;
         end
      end      
      
% get the relevant B-field from the magnet (used later in a test to see if
% all the magnets on the PS are off, or whether at least one of them is
% currently on)
      
      Bsum = abs(BEAMLINE{elemno}.B(PS_index)) ;
      
% if this is a gradient bend, AND we are assigning both its bend and its
% quad to a common power supply, then we need to do things slightly
% differently:

      if ( (strcmp(BEAMLINE{elemno}.Class,'SBEN')) && ...
           (~MultiPS)                                     )
         Bsum = max(abs(BEAMLINE{elemno}.B)) ;
      
         if ( (length(BEAMLINE{elemno}.B) > 1 ) && (Bsum == 0) )
                stat{1} = 0 ;
                stat = AddMessageToStack(stat,...
                    ['Element # ',num2str(elemno), ...
                    ' gradient bend with B == [0 0] in AsignToPS']) ;
         end
         
      end
      Btot = Btot + Bsum ;
  end
  
% if we got this far, abort execution before we do anythig we're going to
% regret

  if (stat{1} == 0)
      return ;
  end
  
% if we are clearing old assignments, do that now and return

  if (psno == 0)
    for count = 1:length(elemlist)
      elemno = elemlist(count) ; psno = oldpslist(count) ;
      if (psno==0)
        continue ;
      end
      BEAMLINE{elemno}.PS(PS_index) = 0 ;
      elist = find(PS(psno).Element==elemno) ;
      PS(psno).Element(elist) = [] ; %#ok<FNDSB>
    end
    return ;
  end
  

  
% if the PS array doesn't go out far enough, extend it

  if (length(PS) < psno)
      PS(psno).Ampl = 1 ;
      PS(psno).SetPt = 1 ;
      PS(psno).Step = 0 ;
      PS(psno).Element = [] ;
      PS(psno).dAmpl = 0 ;
  end
  
% if the magnets are all zero, set the PS Ampl to zero

  if (Btot == 0)
      PS(psno).Ampl = 0 ;
      PS(psno).SetPt = 0 ;
  end
  
% now perform the assignments:  first at the BEAMLINE...

  for count = 1:length(elemlist)
      elemno = elemlist(count) ;
      BEAMLINE{elemno}.PS(PS_index) = psno ;
      if (Btot == 0)
          BEAMLINE{elemno}.B(PS_index) = 1 ;
      end
  end
  
% ...now at the PS

  elist = [PS(psno).Element elemlist'] ;
  elist = sort(elist) ;
  
% eliminate duplicates

  elast = elist(1) ;
  elist2 = elast ;
  for count = 2:length(elist)
      if (elist(count) ~= elast)
          elast = elist(count) ;
          elist2 = [elist2 elast] ;
      end
  end
  
  PS(psno).Element = elist2 ;
      