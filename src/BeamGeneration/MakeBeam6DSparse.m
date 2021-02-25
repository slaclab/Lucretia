function beamtemp = MakeBeam6DSparse( Initial, nsigmas, nzgrid, npgrid )

% MAKEBEAM6DSPARSE Generate a Lucretia beam which emulates the LIAR beam.
%
%    B = MakeBeam6DSparse( Initial, nsigmas, nzgrid, npgrid ) returns a
%    beam in which the longitudinal coordinates are set to a grid defined
%    by nsigmas, nzgrid, and npgrid.  There are 9 rays of equal charge
%    at each grid point in the zP phase plane.  Of the 9 rays at a given
%    grid point, the first point has the transverse coordinates defined
%    in the Initial structure Initial.x.pos/ang and Initial.y.pos/ang
%    fields; the remaining 8 points are at +/- 2 sigmas in x, x', y,
%    and y', respectively, such that the "sigma matrix" for the 9
%    rays at each grid point correspond to the sigma matrix defined by
%    the emittances and Twiss parameters in the Inital data strcture's
%    Twiss and emittance fields.
%
% SEE ALSO:  MakeBeam6DGauss, MakeBeamPZGrid, MakeBeamPZGauss, MakeBeam6DWeighted
%            CreateBlankBeam, InitCondStruc.
%
% Version date:  24-May-2007.

% MOD:
%      24-may-2007, PT:
%         check for bad transverse and total momentum before
%         returning the beam to the caller.
%      PT, 03-nov-2006:
%        bugfix:  incorrect inclusion of correlations when there is an
%        initial dispersion.

%=======================================================================
%
% begin by using the PZgrid tool to generate the beam; since we will
% eventually apply a correlation to the beam, for the time being 
% suppress any initial x/px/y/py values
%
  xpos = Initial.x.pos ;
  ypos = Initial.y.pos ;
  xang = Initial.x.ang ;
  yang = Initial.y.ang ;
  
  Initial.x.pos = 0 ;
  Initial.x.ang = 0 ;
  Initial.y.pos = 0 ;
  Initial.y.ang = 0 ;
  
  beamtemp = MakeBeamPZGrid( Initial, nsigmas, nzgrid, npgrid, 9 ) ;
  ngridpoint = nzgrid * npgrid ;
%
% get the central momentum and the Lorentz factor
%
  p0 = Initial.Momentum ;
  relgam0 = sqrt(p0^2 + (0.000510998918)^2) / 0.000510998918 ;
%
% define a unit matrix
%
  r = eye(6) ;
  offset = zeros(6,ngridpoint*9) ;
%
% set stuff to do the x-plane first
%
  pos = 1 ; ang = 2 ; 
  indxoffset = 1 ;
%
% do the transformations 1 plane at a time
%
  while (pos < 4)
      
      if (pos == 1)
          beta = Initial.x.Twiss.beta ;
          alfa = Initial.x.Twiss.alpha ;
          eta  = Initial.x.Twiss.eta ;
          etap = Initial.x.Twiss.etap ;
          emit = Initial.x.NEmit / relgam0 ;
      else
          beta = Initial.y.Twiss.beta ;
          alfa = Initial.y.Twiss.alpha ;
          eta  = Initial.y.Twiss.eta ;
          etap = Initial.y.Twiss.etap ;
          emit = Initial.y.NEmit / relgam0 ;
      end
      r(ang,pos) = -alfa/beta ;
      r(pos,6) = eta/p0 ;
      r(ang,6) = etap/p0 ; 
      offset(pos,:) = eta*ones(1,ngridpoint*9) ;
      offset(ang,:) = etap*ones(1,ngridpoint*9) ;
      sigma = sqrt(9/2)*sqrt(emit*beta) ; 
      sigp = sqrt(9/2)*sqrt(emit/beta) ;
%
% loop over grid points
%
      for count = 0:ngridpoint-1
%
% set the offset coordinates appropriately
%
          beamtemp.Bunch(1).x(pos,count*9+1+indxoffset) = ...
           beamtemp.Bunch(1).x(pos,count*9+1+indxoffset) + sigma ;
       
          beamtemp.Bunch(1).x(pos,count*9+2+indxoffset) = ...
           beamtemp.Bunch(1).x(pos,count*9+2+indxoffset) - sigma ;
       
          beamtemp.Bunch(1).x(ang,count*9+3+indxoffset) = ...
           beamtemp.Bunch(1).x(ang,count*9+3+indxoffset) + sigp ;
       
          beamtemp.Bunch(1).x(ang,count*9+4+indxoffset) = ...
           beamtemp.Bunch(1).x(ang,count*9+4+indxoffset) - sigp ;

      end
%
% so that takes care of one set of coordinates for bunch 1.  Now we change
% the indexes and do the other set of coordinates.
%
      pos = pos + 2 ;
      ang = ang + 2 ;
      indxoffset = indxoffset + 4 ;
      
  end
%
% apply the correlation to the bunch
%
  beamtemp.Bunch(1).x = r*beamtemp.Bunch(1).x - offset ;
%
% put in the initial x/px/y/py
%
  beamtemp.Bunch(1).x(1,:) = beamtemp.Bunch(1).x(1,:) + xpos ;
  beamtemp.Bunch(1).x(2,:) = beamtemp.Bunch(1).x(2,:) + xang ;
  beamtemp.Bunch(1).x(3,:) = beamtemp.Bunch(1).x(3,:) + ypos ;
  beamtemp.Bunch(1).x(4,:) = beamtemp.Bunch(1).x(4,:) + yang ;
%
% copy the first bunch coordinates to all the rest of the bunches
%
  for count = 2:Initial.NBunch
      beamtemp.Bunch(count).x = beamtemp.Bunch(1).x ;
  end
  
  beamtemp = CheckBeamMomenta(beamtemp) ;