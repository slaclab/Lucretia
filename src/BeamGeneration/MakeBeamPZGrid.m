function beamout = MakeBeamPZGrid( Initial, nsigmas, nzgrid, npgrid, varargin )

% MAKEBEAMPZGRID Generate Lucretia beam on a grid in longitudinal phase space.
%
%    B = MakeBeamPZGrid( Initial, nsigmas, nzgrid, npgrid ) returns a 
%    Lucretia beam structure in which all the rays have the same transverse
%    coordinates but form a grid in the P-z (longitudinal) phase plane.
%    Argument Initial contains all initial conditions such as beam energies,
%    bunch lengths, etc (see function InitCondStruc for details).  Argument
%    nsigmas tells the # of sigmas on which to truncate.  Arguments
%    nzgrid and npgrid are the number of grid lines in z and P, respectively.
%    One ray is generated at each grid point.
%
%    B = MakeBeamPZGrid( Initial, nsigmas, nzgrid, npgrid, nray ) generates
%    nray rays at each grid point, rather than one ray.
%
%    MakeBeamPZGrid assumes that the beam is Gaussian-distributed in both z
%    and P, and generates the charges of the rays accordingly (ie, the ray
%    charges are not equal).  If nray > 1, all the rays at a given grid point
%    have charges which are equal to one another (but potentially different
%    from rays at other grid points).
%
%    SEE ALSO:  MakeBeam6DSparse, MakeBeam6DGauss, MakeBeamPZGauss, MakeBeam6DWeighted
%               CreateBlankBeam, InitCondStruc.
%
% Version date:  24-May-2007.

% MOD:
%      24-may-2007, PT:
%         check for bad transverse and total momentum before
%         returning the beam to the caller.

%============================================================================

%
% very sketchy early version has virtually no exception checking!
%
  if (nargin > 4)
      nrayper = varargin{1} ;
  else
      nrayper = 1 ;
  end
%
% compute the scaling for charge cut-off by the nsigmas argument 
%
  [nsigz,nsigP] = deal(nsigmas) ;
  Qscale = 1/ erf(nsigz/sqrt(2)) / erf(nsigP/sqrt(2)) ;
%
% compute the fractional charge per z grid point
%
  zwidth = 2 * nsigz / nzgrid ;
  zQ = zeros( 1,nzgrid ) ;
  zpos = zeros(1,nzgrid) ;
  for count = 1:nzgrid
      edge1 = -nsigz+zwidth*(count-1) ; edge2 = -nsigz + count*zwidth ;
      zpos(count) = 0.5 * (edge1+edge2) ;
      zQ(count) = 0.5 * (erf(edge2/sqrt(2))-erf(edge1/sqrt(2))) ;
  end
  zpos = zpos * Initial.sigz + Initial.zpos ;
%
% compute the fractional charge per P grid point
%
  pwidth = 2 * nsigP / npgrid ;
  pQ = zeros( 1,npgrid ) ;
  ppos = zeros(1,npgrid) ;
  for count = 1:npgrid
      edge1 = -nsigP+pwidth*(count-1) ; edge2 = -nsigP + count*pwidth ;
      ppos(count) = 0.5 * (edge1+edge2) ;
      pQ(count) = 0.5 * (erf(edge2/sqrt(2))-erf(edge1/sqrt(2))) ;
  end
  ppos = ppos * Initial.SigPUncorrel + Initial.Momentum ;
%
% compute the fractional charge per grid point
%
  zpQ = zeros(nzgrid,npgrid) ;
  for count1 = 1:nzgrid
      for count2 = 1:npgrid 
          zpQ(count1,count2) = Initial.Q * Qscale * zQ(count1) * pQ(count2) / nrayper ; 
      end
  end
%
% get a blank beam with the right number of rays per bunch 
%
  beamtemp = CreateBlankBeam( 1 ,...
                              nrayper*nzgrid*npgrid, ...
                              Initial.Momentum,...
                              Initial.BunchInterval  ) ;
% 
% set the ray coordinates and charge of the bunch 
%
  rayno = 0 ;
  for count1 = 1:nzgrid
      for count2 = 1:npgrid
          for count3 = 1:nrayper
              rayno = rayno + 1 ;
              beamtemp.Bunch.Q(rayno) = zpQ(count1,count2) ;
              beamtemp.Bunch.x(5,rayno) = zpos(count1) ;
              beamtemp.Bunch.x(6,rayno) = ppos(count2) ...
                 + Initial.PZCorrel * (zpos(count1)-Initial.zpos) ;
              beamtemp.Bunch.x(1,rayno) = Initial.x.pos ;
              beamtemp.Bunch.x(2,rayno) = Initial.x.ang ;
              beamtemp.Bunch.x(3,rayno) = Initial.y.pos ;
              beamtemp.Bunch.x(4,rayno) = Initial.y.ang ;
          end
      end
  end
%
% now set the output beam equal to the temp beam
%
  beamout = beamtemp ;
%
% set additional bunches
%
  for count = 1:Initial.NBunch-1
      beamout.Bunch = [beamout.Bunch ; beamtemp.Bunch] ;
  end
  
  beamout = CheckBeamMomenta(beamout) ;
      