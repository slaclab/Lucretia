function stat = CalcAndApplyLossFactor( Initial )
%
% CalcAndApplyLossFactor -- use tracking to estimate the cavity loss factor
% in the tutorial linac lattice, and apply it to the lattice

% During tracking, the mean energy loss due to wakefields is calculated
% directly by applying the wakefields.  During other operations (setting
% the momentum profile, calculation of R matrices and Twiss parameters,
% etc), since there is no tracking performed, we need an alternate method
% to estimate the mean energy loss due to the longitudinal wakefields.
% This is the LCAV Kloss factor, which is in V/C/m.  During non-tracking
% operations, the mean loss is estimated as:
%
%    Loss [V] = Kloss * L_cav [m] * Q_bunch [C].
%
% This function estimates the loss as follows:  it tracks a bunch in order
% to calculate the true energy after tracking, including wake losses; the
% final energy from tracking is compared to the optics final energy to
% estimate the loss factor.  After making this calculation, the loss factor
% is applied to all the cavities and the momentum profile is recalculated
% and the optics scaled.

% Version date:  12-Mar-2008.

% Revision History:
%
%==========================================================================

  global BEAMLINE ; 
  
% we only want 1 bunch, so set that now
  
  Initial.NBunch = 1 ;
  
% generate the simplest beam imaginable for this -- one which has
% longitudinal distribution based on the Initial data structure, but is
% pointlike in the transverse degrees of freedom

  beam0 = MakeBeamPZGrid( Initial, 3, 31, 5 ) ;
  
% find the RF cavities, most importantly the last one, and the # of
% cavities

  cavlist = findcells(BEAMLINE,'Class','LCAV') ;
  lastcav = cavlist(end) ;
  Ncav = length(cavlist) ;
  
% find the expected final momentum from the element after the last cavity

  PfModel = BEAMLINE{lastcav+1}.P ; 
  
% track the beam through the linac

  [stat,beamout] = TrackThru(1,lastcav,beam0,1,1,0) ; 
  if (stat{1} ~= 1)
      return ;
  end
  
% get the final centroid energy of the bunch

  [x,sig] = GetBeamPars(beamout,1) ;
  
% compute the loading in GeV

  loading = PfModel - x(6) ;
  
% the loss factor is the loading per meter per coulomb, in volts

  Kloss = loading*1e9 / Ncav / BEAMLINE{lastcav}.L / Initial.Q ;
  disp(['Change in cavity loading factor:  ',num2str(Kloss,'%e'),' V/C/m']) ;
  
% apply the loading factor and recompute the design momentum profile,
% scaling the magnets as needed.  Note that the assignment below, by using
% a sum rather than a straight assignment, allows iteration -- subsequent
% iterations of this function will produce better and better estimates of
% the loss factor.  Why it should need to be iterated I do not know, but I
% know that it does seem to be useful...

  for count = cavlist
      BEAMLINE{count}.Kloss = BEAMLINE{count}.Kloss + Kloss ;
  end
  
  stat = SetDesignMomentumProfile( 1, length(BEAMLINE), Initial.Q, 5 ) ;
