function iss = UpdateMomentumProfile( istart, iend, Q, P0, varargin )

% UPDATEMOMENTUMPROFILE Update the momentum profile based on present
% klystron complement and RF system values.
%
%   iss = UpdateMomentumProfile( istart, iend, Q, P0 ) computes the present
%      P and Egain values at all elements between istart and iend, and uses
%      these to replace the values currently stored in BEAMLINE.  Q is the
%      design bunch charge in coulombs, P0 is the desired initial momentum
%      in GeV/c.  As part of the process any klystron which has status ==
%      MAKEUP is changed to status == ON, and any which has status ==
%      TRIPPED is changed to status == TRIPSTANDBY.  Return iss is a
%      Lucretia status cell array (type help LucretiaStatus for more
%      information).
%
%   iss = UpdateMomentumProfile( istart, iend, Q, P0, ScalePS ) applies the
%      new P and Egain values, and optionally scales the power supplies of
%      magnets between istart and iend based on the change in design
%      momentum at the magnets.  If ScalePS == 1, scaling is performed,
%      otherwise it is not.  Magnets without power supplies are not scaled.
%      If several magnets are powered in series by a single supply, the
%      scale factor will be the average of the scalings needed by the
%      magnets in question.  UpdateMomentumProfile does not attempt to
%      scale the klystrons of TCAVs.
%
%   iss = UpdateMomentumProfile( istart, iend, Q, P0, ScalePS, Pf ) scales
%      the computed Egain values such that the final momentum on the
%      downstream face of iend is Pf.  
%
%   Return status:  +1 if all operations were successful, -1 if a power
%   supply was detected which supports magnets outside of the desired
%   range, -2 if a klystron was detected which supports RF units outside
%   the desired range, -3 if the updated momentum profile causes P<0 at one
%   or more points within the range, -4 if invalid range, Q or P0
%   arguments, -5 if an error occurred while attempting to scale momentum,
%   -6 if an error related to synchrotron radiation loss calculations
%   occurs.  If status ~= +1, no change to the momentum profile or the
%   magnet strengths will be implemented.
%
% See also ComputeMomentumProfile, SetDesignMomentumProfile.
%
% Version date:  02-may-2006.

% MOD:
%      17-may-2010, GW:
%         add ,1 in PSTrim for Floodland and minor fixes to make green box happy
%      02-may-2006, PT:
%         bugfix -- egain computed from the returns of 
%         ComputeMomentumProfile is in GeV, but Egain field of LCAV is in
%         MeV.
%      12-jan-2006, PT:
%         Changes in support of synchrotron radiation:  mainly differences
%         in the use of ComputeMomentumProfile.

%==========================================================================

  global BEAMLINE ;
  global KLYSTRON ; %#ok<NUSED>
  global PS ;
  iss = InitializeMessageStack( ) ;
  step = 1 ;
% 
% verify correct ranges for all variables 
%
  if ( (istart < 1) || (istart > length(BEAMLINE)) || ...
       (iend < 1)  ||  (iend > length(BEAMLINE)  ) || ...
       (istart > iend)                            || ...
       (Q<0)                                      || ...
       (P0<=0)                                          )
     iss = AddMessageToStack(iss, ...
       'Invalid arguments specified for UpdateMomentumProfile') ;
     iss{1} = -4 ;
     return ;
  end
%  
% get the klystrons and power supplies which support elements within the
% range of BEAMLINE
%
  [statcall,klist] = GetKlystronsInRange( istart, iend ) ;
  if (statcall{1} ~= 1)
      iss{1} = -2 ;
      iss = AddStackToStack(iss,statcall) ;
      return ;
  end
  plist = [] ;
  if (nargin > 4)
      scale = varargin{1} ;
      if (scale==1)
          [statcall,plist] = GetPSInRange( istart, iend ) ;
          if (statcall{1} ~= 1)
            iss{1} = -1 ;
            iss = AddStackToStack(iss,statcall) ;
            return ;
          end
      end
  end
%
% do we want to fix the final momentum?
%
  if (nargin > 5)
      Pf = varargin{2} ;
      if (Pf <= 0)
          iss{1} = -4 ;
          iss = AddMessageToStack(iss,...
            'Invalid final momentum in UpdateMomentumProfile') ;
          return ;
      end
  else
      Pf = 0 ;
  end
%
% update the klystron status
%
  UpdateKlystronStatus( klist ) ;
%
% compute the current momentum profile
%
  [statcall,V_on,V_off,V_load,V_SR] = ...
      ComputeMomentumProfile( istart, iend, Q ) ;
  if ( (statcall{1}==0) || (statcall{1} == -6) )
      iss{1} = -6 ;
      iss = AddStackToStack(iss,statcall) ;
      return ;
  end 
  total_energy_loss = sum(V_load) + sum(V_SR) ;
  total_energy_gain = sum(V_on(:,1)) - total_energy_loss ;
  pf_model = P0 + total_energy_gain ; 
%
% if scaling is desired, do it now; note that only the voltage can be
% scaled up, since loading and SR are assumed to be un-scalable.  For SLAC
% people, final_scale is the LEM fudge factor.
%
  if (Pf ~= 0)
    final_scale = (Pf - P0 - total_energy_loss) / ...
                  (pf_model - P0 - total_energy_loss) ;
    if (final_scale < 0)
        iss{1} = -5 ; 
        iss = AddMessageToStack(iss,...
            'Negative scale factor required in UpdateMomentumProfile') ;
        return ;
    end
    V_on = V_on * final_scale ;
  end
%
% now compute the P and Egain vectors
%
  Egain = V_on(:,1) - V_load - V_SR ;
  P = P0 + cumsum(Egain) ; P = [P0 ; P] ;
  pm = min(P) ;
  if (pm <=0)
      iss{1} = -3 ;
      iss = AddMessageToStack(iss,'Momentum < 0 detected in UpdateMomentumProfile!') ;
      return ;
  end
%
% loop over power supplies and scale them
%
  if (~isempty(plist))
      for count = 1:length(plist) 
      
          scalemean = 0 ;
      
          for count2 = 1:length(PS(plist(count)).Element)
          
              eptr = PS(plist(count)).Element(count2) ;
              pptr = step*(eptr - istart) + 1 ;
              scalemean = scalemean + P(pptr) / BEAMLINE{eptr}.P ;
          
          end
          scalemean = scalemean / length(PS(plist(count)).Element) ;
          PS(plist(count)).SetPt = scalemean * PS(plist(count)).Ampl ;
      
      end
      statcall = PSTrim(plist,1) ;
      iss = AddStackToStack(iss,statcall) ;
      if (statcall{1} ~= 1)
          iss{1} = statcall{1} ;
      end
  end
%
% now loop over elements and set new P / Egain values
%
  pptr = 0 ;
  for count = istart:step:iend
      pptr = pptr + 1 ;
      BEAMLINE{count}.P = P(pptr) ;
%      if ( Egain(pptr) ~= 0 )
      if ( isfield(BEAMLINE{count},'Egain') )
          BEAMLINE{count}.Egain = Egain(pptr) * 1000 ;
      end
  end
%      