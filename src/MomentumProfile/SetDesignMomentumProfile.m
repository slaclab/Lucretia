function stat = SetDesignMomentumProfile( istart, iend, Q, P0, varargin )
%
% SETDESIGNMOMENTUMPROFILE Compute the design momentum profile and scale
% beamline components to it.
%
% stat = SetDesignMomentumProfile( istart, iend, Q, P0 ) computes the design
%    momentum profile and applies it.  All power supplies and klystrons
%    supporting elements in the range istart:iend inclusive are renormalized,
%    and the status of all klystrons in that range is updated.  All magnets
%    and TCAVs within the range are scaled from the current momentum
%    profile to the new one.
%
% stat = SetDesignMomentumProfile( istart, iend, Q, P0, Pf ) forces the
%    design momentum on the downstream face of BEAMLINE{iend} to be equal
%    to Pf.  This is achieved by scaling the RF structure voltages.  All
%    structure voltages within the range will be scaled, including
%    structures which are currently accelerating the beam, structures
%    powered by klystrons on standby, and structures powered by tripped
%    klystrons.
%
% Return variable stat is a Lucretia status and message cell array (type
%   help LucretiaStatus for more information).  Return values include
%   stat{1} == +1 for full success,  -1 if a power supply was detected
%   which supports magnets outside of the desired range, -2 if a klystron
%   was detected which supports RF units outside the desired range, -3 if
%   the updated momentum profile causes P<0 at one or more points within
%   the range, -4 if invalid range, Q or P0 arguments, -5 if an error
%   occurred while attempting to scale momentum.  If status ~= +1, no
%   change to the momentum profile or the element parameters will be
%   implemented.
%
% See also:  UpdateMomentumProfile, UpdateKlystronStatus, RenormalizePS,
%    RenormalizeKlystron.
%
% Version date:  12-May-2006

% MOD:
%       24-Sep-2012, GW:
%          don't error if no PS's/KLYSTRON's
%       12-may-2006, PT:
%          bugfix in units of Egain field (MeV, not GeV).
%       13-jan-2006, PT:
%          scaling respects Slices.
%       06-dec-2005, PT:
%          changes to support synchrotron radiation, and also to support
%          new in/out arglist of ComputeMomentumProfile (new list is also
%          in support of synchrotron radiation).
%       30-sep-2005, PT:
%           Support for TCAVs.

%==========================================================================

global BEAMLINE
stat = InitializeMessageStack( ) ;
%
% verify correct ranges for all variables
%
if ( (istart < 1) || (istart > length(BEAMLINE)) || ...
    (iend < 1)  ||  (iend > length(BEAMLINE)  ) || ...
    (istart > iend)                            || ...
    (Q<0)                                      || ...
    (P0<=0)                                          )
  stat = AddMessageToStack(stat, ...
    'Invalid arguments specified for SetDesignMomentumProfile') ;
  stat{1} = -4 ;
  return ;
end
if (nargin == 5)
  Pfdes = varargin{1} ;
  if (Pfdes <= 0)
    stat = AddMessageToStack(stat,...
      'Invalid final momentum specified for SetDesignMomentumProfile') ;
    stat{1} = -4 ;
    return
  end
else
  Pfdes = 0 ;
end
%
% get the klystrons and power supplies which support elements within the
% range of BEAMLINE
%
[~,klist] = GetKlystronsInRange( istart, iend ) ;
% if (statcall{1} ~= 1)
%   stat{1} = -2 ;
%   stat = AddStackToStack(stat,statcall) ;
%   %       return ;
% end
[~,plist] = GetPSInRange( istart, iend ) ;
% if (statcall{1} ~= 1)
%   stat{1} = -1 ;
%   stat = AddStackToStack(stat,statcall) ;
%   return ;
% end
%
% compute the momentum profile without scaling
%
[statcall,V_on,~,V_load,V_SR,P] = ...
  ComputeMomentumProfile( istart, iend, Q, P0 ) ;
if (statcall{1} ~= 1)
  stat{1} = statcall{1} ;
  stat = AddStackToStack(stat,statcall) ;
  return ;
end
Pfact = P(length(P)) ;
if (Pfdes == 0)
  Pfdes = Pfact ;
end
%
% momentum scaling:
%
if ( Pfdes ~= Pfact )
  sum_V_on = sum(V_on(:,1)) ;
  %
  % we can only _specify_ a final momentum if the on-beam energy gain is not
  % zero, so check that now
  %
  if (sum_V_on==0)
    stat{1} = -5 ;
    stat = AddMessageToStack(stat, ...
      'Final momentum specification invalid when on-beam energy gain zero') ;
    return ;
  end
  %
  % we need to scale V_on until it gets us the right final momentum, taking
  % into account the loading and SR losses
  %
  V_needed = Pfdes - P0 + sum(V_load) + sum(V_SR) ;
  V_scale = V_needed / sum_V_on ;
  
else
  
  V_scale = 1.0 ;
  
end
%
% at this point, we're basically guaranteed success so we can proceed to do
% things which actually change the lattice:
%
% update the klystron status
%
if ~isempty(klist)
  UpdateKlystronStatus( klist ) ;
end
%
% renormalize klystrons and power supplies
%
if ~isempty(klist)
  for count = klist
    RenormalizeKlystron( count ) ;
  end
end
if ~isempty(plist)
  for count = plist
    RenormalizePS( count ) ;
  end
end
%
% do voltage scaling
%
list = findcells(BEAMLINE,'Class','LCAV',istart,iend) ;
for count = list
  BEAMLINE{count}.Volt = BEAMLINE{count}.Volt * V_scale ;
end
%
% recompute the momentum profile using the new voltages
%
[~,V_on,~,V_load,~,P] = ...
  ComputeMomentumProfile( istart, iend, Q, P0 ) ;
%
% apply the profile and scale magnets appropriately, taking care that
% elements which are "slices" of a single physical element get the correct
% scaling
%
dcount = istart-1 ;
for count = istart:iend
  count2 = count-dcount ;
  if (isfield(BEAMLINE{count},'Slices'))
    slice1 = BEAMLINE{count}.Slices(1) ;
    slice2 = BEAMLINE{count}.Slices(end)+1 ;
    if (slice1 == count)
      Pmean = (P(slice1-dcount)+P(slice2-dcount))/2 ;
      Pold = BEAMLINE{slice1}.P ;
    end
  else
    Pmean = P(count2) ;
    Pold = BEAMLINE{count}.P ;
  end
  if (isfield(BEAMLINE{count},'B'))
    BEAMLINE{count}.B = BEAMLINE{count}.B * ...
      Pmean / Pold ;
  end
  if (strcmp(BEAMLINE{count}.Class,'TCAV'))
    BEAMLINE{count}.Volt = BEAMLINE{count}.Volt * ...
      Pmean / Pold ;
  end
  if (isfield(BEAMLINE{count},'Egain'))
    BEAMLINE{count}.Egain = 1000*(V_on(count2,1)-V_load(count2)) ;
  end
  BEAMLINE{count}.P = P(count2) ;
end
%