function stat = AssignToKlystron( elemlist, klysnum )

% ASSIGNTOKLYSTRON Assign elements to a klystron.
%
%   stat = AssignToKlystron( element_list, klystron_number ) assigns the
%      BEAMLINE elements in the element_list to the KLYSTRON indexed by
%      klystron_number.  The elements are added to any existing ones which
%      are already assigned to the KLYSTRON.  If klysnum is greater than
%      the present length of the KLYSTRON array, it will be extended; if
%      the KLYSTRON array is missing, it will be created; if
%      klystron_number == 0, the current klystron assignments for the
%      elements in element_list will be deleted. Return argument stat is a
%      cell array, with stat{1} the status and stat{2...} text messages
%      from the function.  The value of stat{1} == 1 for error-free
%      execution, or == 0 if error occurs (non-RF device in the element
%      list and/or element in the element list which already has a
%      klystron).  In the event of error, no assignments will be performed.

% MOD:
%       30-sep-2005, PT:
%          allow TCAVs to be assigned to a klystron.

%==========================================================================

stat = InitializeMessageStack( ) ;
global BEAMLINE ;
global KLYSTRON ;

elemlist=elemlist(:)';

% make sure that the new devices are not yet assigned to klystrons, and are
% valid KLYSTRON devices

Vtot = 0 ;
elemlist = elemlist(:) ;
oldklyslist = [] ;
for count = 1:length(elemlist)
  elemno = elemlist(count) ;
  if (~strcmp(BEAMLINE{elemno}.Class,'LCAV') & ...
      ~strcmp(BEAMLINE{elemno}.Class,'TCAV')       )
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      ['Non-RF element # ',num2str(elemno),...
      ' detected in AssignToKlystron input']) ;
  else
    if (isfield(BEAMLINE{elemno},'Klystron'))
      if ( (~isempty(BEAMLINE{elemno}.Klystron)) & ...
          (BEAMLINE{elemno}.Klystron ~= 0) & ...
          (BEAMLINE{elemno}.Klystron ~= klysnum) & ...
          (klysnum ~= 0)                               )
        stat{1} = 0 ;
        stat = AddMessageToStack(stat,...
          ['Element # ',num2str(elemno), ...
          ' already has conflicting klystron assignment in ',...
          'AssignToKlystron']) ;
      end
      
      % if the klystron number == 0 and this element has a klystron assignment,
      % capture the existing assignment
      
      if (klysnum==0)
        if (isempty(BEAMLINE{elemno}.Klystron))
          oldklyslist = [oldklyslist 0] ;
        else
          oldklyslist = [oldklyslist BEAMLINE{elemno}.Klystron] ;
        end
      end
    else
      if (klysnum == 0)
        oldklyslist = [oldklyslist 0] ;
      end
    end
  end
  Vtot = Vtot + abs(BEAMLINE{elemno}.Volt) ;
end

% if we got this far, abort execution before we do anything we're going to
% regret

if (stat{1} == 0)
  return ;
end

% if the klystron number == 0, we need to clear the assignment from the
% KLYSTRON array.  Do that now, and return

if (klysnum == 0)
  for count = 1:length(elemlist)
    elemno = elemlist(count) ; klysno = oldklyslist(count) ;
    if (klysno==0)
      continue ;
    end
    BEAMLINE{elemno}.Klystron = 0 ;
    elist = find(KLYSTRON(klysno).Element==elemno) ;
    Klystron(klysno).Element(elist) = [] ;
  end
  return ;
end

% if the klystron doesn't go out far enough, extend it

if (length(KLYSTRON) < klysnum)
  KLYSTRON(klysnum).Ampl = 1 ;
  KLYSTRON(klysnum).AmplSetPt = 1 ;
  KLYSTRON(klysnum).AmplStep = 0 ;
  KLYSTRON(klysnum).Phase = 0 ;
  KLYSTRON(klysnum).PhaseSetPt = 0 ;
  KLYSTRON(klysnum).PhaseStep = 0 ;
  KLYSTRON(klysnum).Element = [] ;
  KLYSTRON(klysnum).dAmpl = 0 ;
  KLYSTRON(klysnum).dPhase = 0 ;
  KLYSTRON(klysnum).Stat = 'ON' ;
end

% if all the RF stations are at zero volts, then assign the klystron to
% have zero amplitude and the RF stations to have unit amplitude

if (Vtot == 0)
  KLYSTRON(klysnum).Ampl = 0 ;
  KLYSTRON(klysnum).AmplSetPt = 0 ;
end

% now perform the assignments:  first at the BEAMLINE...

for count = 1:length(elemlist)
  elemno = elemlist(count) ;
  BEAMLINE{elemno}.Klystron = klysnum ;
  if (KLYSTRON(klysnum).Ampl == 0)
    BEAMLINE{elemno}.Volt = 1 ;
  end
end

% ...now at the klystron

if (klysnum > 0)
  elist = [KLYSTRON(klysnum).Element elemlist] ;
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
  
  KLYSTRON(klysnum).Element = elist2 ;
  
end
