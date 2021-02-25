function B = SBendStruc( L, BField, Angle, E, H, ...
                         Hgap, Fint, Tilt, Name )
%
% SBENDSTRUC Return the Lucretia data structure for a sector bend magnet.
%
% B = SBendStruc( length, BField, Angle, E, H, Hgap, Fint, Tilt, Name )
%    returns a properly formed Lucretia data structure for a sector bend
%    magnet.  Calling arguments are as follows:
%
%    L:       numeric, design arc length in m.
%    BField:  either 1 x 1 numeric (bend strength in T.m) or
%            1 x 2 numeric (bend strength followed by integrated
%            gradient in T).
%    Angle:   design bend angle in radians.
%    E:    pole face rotation wrt the nominal beamline (radians).
%    H:    pole face curvature (1/m).
%    Hgap: half-gap of magnet (m).
%    Fint: fringe field integral value (-).
%    Tilt:    design xy rotation of bend magnet (radians).
%    Name:    bend magnet name (char array).
%
%    Note:  parameters E, H, Hgap, and Fint can be either 1 x 1 or 1 x 2.
%    If 1 x 1, then the single parameter value will be used for both the
%    entry end exit faces of the magnet; if 1 x 2, then the first value
%    will be used at the entry face and the second value will be used at
%    the exit face.
%

% MOD:
%       28-sep-2005, PT:
%          Make dB a scalar by default.

%================================================================

B.Name = Name ;
B.S = 0 ; B.P = 0 ;
B.Class = 'SBEN' ;
B.L = L ;
if (length(BField) == 1)
    B.B = BField ;
else
    B.B = [BField(1) BField(2)] ;
end
B.dB = 0  ;
B.Angle = Angle ;
if (length(E) == 1)
    B.EdgeAngle = [E E] ;
else
    B.EdgeAngle = [E(1) E(2)] ;
end
if (length(Hgap) == 1)
    B.HGAP = [Hgap Hgap] ;
else
    B.HGAP = [Hgap(1) Hgap(2)] ;
end
if (length(Fint) == 1)
    B.FINT = [Fint Fint] ;
else
    B.FINT = [Fint(1) Fint(2)] ;
end
if (length(H) == 1)
    B.EdgeCurvature = [H H] ;
else
    B.EdgeCurvature = [H(1) H(2)] ;
end
B.Tilt = Tilt ; 
B.PS = 0 ; B.Offset = [0 0 0 0 0 0] ;
B.Girder = 0 ;

TrackFlag.SynRad = 0 ;
B.TrackFlag = TrackFlag ;
B.TrackFlag.LorentzDelay = 0 ;
B.TrackFlag.Aper = 0 ;
