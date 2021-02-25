function [stat, V_on, V_off, V_load, V_SR, varargout] = ...
       ComputeMomentumProfile( istart, iend, Q, varargin )

% COMPUTEMOMENTUMPROFILE Compute the design/expected momentum along a
% beamline.
%
%   [stat, V_on, V_off, V_load, V_SR] = ComputeMomentumProfile( istart,
%      iend, Q ) loops over the BEAMLINE from istart to iend and returns 4
%      real matrices.  Matrix V_on is the cosine-like, sine-like, and total      
%      voltage for all structures which are on the beamline (all structures
%      with klystron status == ON or MAKEUP).  V_off is the same for all
%      structures with klystron status == STANDBY.  V_load is the beam
%      loading voltage for a bunch with charge Q.  V_SR is the energy loss
%      from synchrotron radiation in bends, correctors, or TCAVs which have
%      the appropriate tracking flag set. Return argument stat is a
%      Lucretia status cell array (type help LucretiaStatus for more
%      information).
%
%    [..., P] = ComputeMomentumProfile( ... , P0 ) returns as a final
%       argument a vector of design P values for each element in the
%       region, given initial momentum P0.  P has 1 entry more than the
%       number of elements between istart and iend, inclusive; the last
%       element in P holds the final momentum.
%
% Return values V_on, V_off, V_load, and V_SR are in GeV; P is in GeV/c.
%
% Return status:  +1 for success, -4 if invalid values of istart, iend, Q,
%    P0, Pf supplied, 0 if invalid synchrotron radiation parameters are
%    encountered, -6 if unable to compute SR losses in some elements due to
%    negative or zero incoming momentum in those elements.
%
% See also:  UpdateMomentumProfile, SetDesignMomentumProfile.
%
% Version date:  11-May-2006.

% MOD:
%       11-may-2006, PT:
%           bugfix:  beam loading for LCAVs was not being put into the
%           correct data structure for return to calling routines.
%       06-dec-2005, PT:
%           support for SR.  Eliminate return of dP, change V_on, V_off,
%           V_load to 2-d matrices, eliminate support for scaling (move
%           scaling to higher-level apps which call
%           ComputeMomentumProfile).
%       30-sep-2005, PT:
%           Support for TCAVs.

%==========================================================================

  global BEAMLINE ;
  global KLYSTRON ; 
  varargout{1}=[];
  stat = InitializeMessageStack( ) ;
  P0 = 0 ;
  IssuedBadSRMsg = 0 ;
  nelm = abs(iend-istart)+1 ;
  step = 1 ;
  if ( (istart < 1) | (istart > length(BEAMLINE)) )
      stat = AddMessageToStack(stat, ...
          'Invalid start element specified for ComputeMomentumProfile') ;
      stat{1} = -4 ;
  end
  if ( (iend < 1) | (iend > length(BEAMLINE)) )
      stat = AddMessageToStack(stat, ...
          'Invalid final element specified for ComputeMomentumProfile') ;
      stat{1} = -4 ;
  end      
  if ( istart > iend )
    stat{1} = -4 ;
    stat = AddMessageToStack(stat, ...
        'Invalid start-finish relationship for ComputeMomentumProfile') ;
  end
%
% optional argument stuff
%
  if (nargin >= 4)
      P0 = varargin{1} ;
      if (P0 <= 0)
        stat = AddMessageToStack(stat, ...
          'Invalid initial momentum specified for ComputeMomentumProfile') ;
        stat{1} = -4 ;
      end
  else
      P0 = BEAMLINE{istart}.P ;
  end
%
% not optional stuff
%
  P = zeros(nelm+1,1) ;
  V_on = zeros(nelm,3) ;
  V_off = zeros(nelm,3) ;
  V_load = zeros(nelm,1) ;
  V_SR = zeros(nelm,1) ;
  Vcos = 0 ; Vsin = 0 ; V = 0 ; vload = 0 ; 
%
% have we already failed?
%
  if (stat{1} ~= 1)
      return ;
  end
%
% loop over elements
%
  eptr = 0 ; P_elemstart = P0 ;
  for count = istart:step:iend
      Do_SR = 0 ;
      if (isfield(BEAMLINE{count},'TrackFlag'))
       if (isfield(BEAMLINE{count}.TrackFlag,'SynRad'))
        if (BEAMLINE{count}.TrackFlag.SynRad > 0)
         if (P_elemstart > 0)
          Do_SR = 1 ;
         else
          if (IssuedBadSRMsg == 0)
            stat{1} = -6 ;
            stat = AddMessageToStack(stat,...
                ['Encountered P<=0 in element ',num2str(count),...
                ', SR calcs disabled downstream of this element']) ;
            IssuedBadSRMsg = 1 ;
          end
         end
        end
       end
      end
      eptr = eptr + 1 ;
      P(eptr) = P_elemstart ;
      if ( (strcmp(BEAMLINE{count}.Class,'LCAV')) | ...
           (strcmp(BEAMLINE{count}.Class,'TCAV'))       )
          phi = BEAMLINE{count}.Phase ; V = BEAMLINE{count}.Volt/1000 ; % deg and GV/c
          K = BEAMLINE{count}.Klystron ;
          if (K > 0)
              phi = phi + KLYSTRON(K).Phase ;
              V = V * KLYSTRON(K).Ampl ;
              if ( (strcmp(KLYSTRON(K).Stat,'ON'    )) |...
                   (strcmp(KLYSTRON(K).Stat,'MAKEUP'))      )
                kstat = 1 ;
              elseif (strcmp(KLYSTRON(K).Stat,'STANDBY'))
                kstat = 2 ;
              else
                kstat = 0 ;
              end
          end
          Vcos = V * cos(pi*phi/180) ; Vsin = V * sin(pi*phi/180) ;
      end
      if (strcmp(BEAMLINE{count}.Class,'LCAV'))
          vload = Q * BEAMLINE{count}.L * BEAMLINE{count}.Kloss ; % volts
          V_load(eptr) =  vload / 1e9 ;
          if ( (K <= 0) | (kstat == 1) )
              V_on(eptr,1) =  Vcos ;
              V_on(eptr,2) =  Vsin ;
              V_on(eptr,3) =  V ;
          elseif (kstat == 2)
              V_off(eptr,1) = Vcos ;
              V_off(eptr,2) = Vsin ;
              V_off(eptr,3) = V ;
          end
      end
      if (strcmp(BEAMLINE{count}.Class,'TCAV'))
          vload = Q * BEAMLINE{count}.L * BEAMLINE{count}.Kloss ; % volts
          V_load(eptr) = vload / 1e9 ;
      end
      
% now for some elements which are only used if they have SR supported

      if (Do_SR == 0)
          P_elemstart = P_elemstart + V_on(eptr,1) - V_load(eptr) ;
          continue ;
      end
      
      Lrad = 0 ; Brad = 0 ;
      if ( (strcmp(BEAMLINE{count}.Class,'XCOR')) | ...
           (strcmp(BEAMLINE{count}.Class,'YCOR')) | ...   
           (strcmp(BEAMLINE{count}.Class,'MULT')) )
         if (isfield(BEAMLINE{count},'Lrad'))
            Lrad = BEAMLINE{count}.Lrad ;
         end
         if (Lrad <= 0)
            Lrad = BEAMLINE{count}.L ;
         end
      end
      if ( (strcmp(BEAMLINE{count}.Class,'SBEN')) | ...
           (strcmp(BEAMLINE{count}.Class,'TCAV'))       )
         Lrad = BEAMLINE{count}.L ;
      end

      if ( (strcmp(BEAMLINE{count}.Class,'XCOR')) | ...
           (strcmp(BEAMLINE{count}.Class,'YCOR')) | ...
           (strcmp(BEAMLINE{count}.Class,'SBEN'))       )
         Brad = BEAMLINE{count}.B(1) ;
      end
      if (strcmp(BEAMLINE{count}.Class,'TCAV'))
         if (kstat==1)
           Brad = Vcos / 0.3 ;
         else
           Brad = 0 ;
         end
      end
      if (strcmp(BEAMLINE{count}.Class,'MULT'))
         i0 = find(BEAMLINE{count}.PoleIndex==0) ;
         by = BEAMLINE{count}.B(i0) .* cos(BEAMLINE{count}.Tilt(i0)) ;
         bx = BEAMLINE{count}.B(i0) .* sin(BEAMLINE{count}.Tilt(i0)) ;
         Brad = sqrt(bx^2 + by^2) ;
      end
      if Lrad>0
        [stat,V_SR(eptr)] = CalculateSR(P_elemstart, Brad, Lrad) ;
        if (stat{1}==0)
           stat = AddMessageToStack(stat,...
                ['Invalid SR Options, element ',num2str(count)]) ;
           return ;
        end
      end
      P_elemstart = P_elemstart - V_load(eptr) - V_SR(eptr) ;
  end
%
% put in the final momentum
%  
  P(nelm+1) = P_elemstart ;
%
% optionally return the vector P
%
  if (nargout > 5)
    varargout{1} = P ;
  end
%
% and that's it.
%
          
          