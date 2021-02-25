function S = CorrectorStruc( L, B, tilt, type, name )

% CORRECTORSTRUC return Lucretia data structure for a dipole steering
% corrector.
%
%   S = CorrectorStruc( L, B, tilt, type, name ) returns a correct
%      Lucretia data structure for a dipole corrector with length L,
%      integrated field B, rotation angle tilt.  If type==1 it is
%      returned as an XCOR; if type==2, it is returned as a YCOR;
%      if type==3, it is returned as an XYCOR, a single beamline 
%      device with both horizontal and vertical correction windings.
%      If the device is an XYCOR, then input argument B should be
%      1 x 2, otherwise it should be scalar. Lucretia will assume 
%      that L is in meters, B in T.m, tilt in radians.
%
% Version date:  21-May-2007.

% MOD:
%      21-may-2007, PT:
%         support for XYCORs.

%=====================================================================

S.Name = name ;
S.S = 0 ; S.P = 0 ;
switch type
  case 1
    S.Class = 'XCOR' ;
  case 2
    S.Class = 'YCOR' ;
  otherwise
    S.Class = 'XYCOR' ;
    type = 3 ;
end
S.L = L ; S.B = B ; 
if (type == 3)
  S.dB = [0 0] ;
  S.PS = [0 0] ;
else
  S.dB = 0 ;
  S.PS = 0 ;
end
S.Tilt = tilt ; 
S.Offset = [0 0 0 0 0 0] ; S.Girder = 0 ;
S.Lrad = L ;

S.TrackFlag.SynRad = 0 ;
S.TrackFlag.ZMotion = 0 ;
S.TrackFlag.LorentzDelay = 0 ;