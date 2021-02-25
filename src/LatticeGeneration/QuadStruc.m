function Q = QuadStruc( length, IGdL, Tilt, aper, Name )
%
% QUADSTRUC Return a Lucretia data structure for a quadrupole magnet
%
% Q = QuadStruc( L, B, Tilt, aper, Name ) returns a fully-formed Lucretia
%    data structure for a quadrupole, with the fields specified in the 5
%    calling arguments (length, integrated gradient, xy rotation, name)
%    filled in and other fields in a default state.  The magnetic field
%    convention is:
%
%       Q.B = B'L = K1L * brho, B(r) = Q.B * r / Q.L.
%

%================================================================

Q.Name = Name ;
Q.S = 0 ; Q.P = 0 ;
Q.Class = 'QUAD' ;
Q.L = length ;
Q.B = IGdL ; Q.dB = 0 ;
Q.Tilt = Tilt ; 
Q.aper = aper ;
Q.PS = 0 ; Q.Offset = [0 0 0 0 0 0] ;
Q.Girder = 0 ;

TrackFlag.SynRad = 0 ;
Q.TrackFlag = TrackFlag ;
Q.TrackFlag.ZMotion = 0 ;
Q.TrackFlag.LorentzDelay = 0 ;
Q.TrackFlag.Aper = 0 ;
