function S = RFStruc( L, V, phi, freq, lfile, tfile, ...
                      eloss, aper, name, varargin )
%
% RFSTRUC return a Lucretia data structure for an RF accelerating
% or deflecting structure.
%
%   S = RFStruc( L, V, phi, freq, lfile, tfile, eloss, aper, name ) 
%      returns a structure for a linac accelerating element.  Argument
%      units are meters (L,aper), MV (V), rad/2pi (phi), MHz (freq), V/C
%      (eloss).  lfile and tfile are index numbers of wakefield files.
%      Note that in the returned data structure the phase will be converted
%      to degrees and the loss parameter to V/C/m.  The sign convention for
%      the phase is that a positive phase puts the RF crest ahead of the
%      bunch centroid (ie, over-accelerates the head wrt the tail), thus
%      the energy gain for a particle with 5th coordinate z, where z<0 ==
%      bunch head, is:
%
%          dE = V cos (2pi freq/c z + phi).
%
%      By default, RFStruc returns an RF structure with no HOM BPMs.  Use
%      AddBPMToLcav to add HOM BPMs.
%
%   S = RFStruc( L, V, phi, freq, lfile, tfile, eloss, aper, name, mode )
%      returns either an accelerating structure (mode==0, default) or a
%      deflecting structure (mode==1).  The Transverse structure produces a
%      horizontal deflection given by:
%
%          dPx = V/P cos(2pi freq/c z + phi).
%
% Version date:  12-Feb-2007.

% MOD:
%       12-Feb-2007, PT:
%          SynRad tracking flag for TCAV.
%       30-sep-2005, PT:
%          return TCAV if requested!

%========================================================================

if (nargin==9)
  mode = 0 ;
end
if (nargin>10)
  error('Invalid argument list for RFStruc.m') ;
end
if (nargin==10)
  if (varargin{1} == 1)
    mode = 1 ;
  else
    mode = 0 ;
  end
end
S.Name = name ;
S.S = 0 ; S.P = 0 ; S.Egain = 0 ;
S.Class = 'LCAV' ;
if (mode==1)
    S.Class = 'TCAV' ;
end
S.L = L ; 
S.Volt = V ; S.Phase = phi*360 ; S.Freq = freq ;
S.Kloss = eloss / L ; 
S.dV = 0 ;S.dPhase = 0 ;
S.Offset = [0 0 0 0 0 0] ;
S.aper = aper ;
if (S.aper == 0)
    S.aper = 1 ;
end
if (mode==1)
    S.Tilt = 0 ;
end
S.Girder = 0 ;
S.Klystron = 0 ;

S.NBPM = 0 ;
S.BPMOffset = [ ] ;
S.BPMResolution = 0 ;

S.Wakes = [lfile tfile] ;
S.TrackFlag.SRWF_Z = min(1,lfile) ;
S.TrackFlag.SRWF_T = min(1,tfile) ;
S.TrackFlag.LRWF_T = 0 ;
S.TrackFlag.LRWF_ERR = 0 ;
S.TrackFlag.ZMotion = 0 ;
S.TrackFlag.LorentzDelay = 0 ;
S.TrackFlag.Aper = 0 ;
S.TrackFlag.GetSBPMData = 0 ;
if (mode==1)
    S.TrackFlag.SynRad = 0 ;
end
